use std::{collections::HashMap, process};
use tokio::sync::Mutex;
use tokio_sqlite::Connection;

pub struct DataStorage {
    connection: Mutex<Connection>,
    cache: Mutex<HashMap<String, (i32, String, chrono::NaiveDateTime, chrono::NaiveDateTime)>>,
}

impl DataStorage {
    pub async fn new(db_path: &str) -> Result<&'static Self, tokio_sqlite::Error> {
        match std::fs::create_dir_all(std::path::Path::new(db_path).parent().unwrap()) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to create database directory: {}", e);
                process::exit(1);
            }
        }
        let mut connection = Connection::open(db_path).await?;
        if let Err(e) = connection.execute(
            "CREATE TABLE IF NOT EXISTS authorizations (
    uid INTEGER,
    path TEXT,
    token TEXT PRIMARY KEY,
    expiration TIMESTAMP
)",
            [],
        ).await {
            return Err(e);
        }
        if let Err(e) = connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_uid_path ON authorizations (uid, path)",
            [],
        ).await {
            return Err(e);
        }
        if let Err(e) = connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_expiration ON authorizations (expiration)",
            [],
        ).await {
            return Err(e);
        }
        let cache = Box::leak(Box::new(DataStorage {
            connection: Mutex::new(connection),
            cache: Mutex::new(HashMap::new()),
        }));

        let ref2 = &*cache;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                let _ = ref2.clean().await;
            }
        });

        Ok(cache)
    }

    async fn clean(&self) -> Result<(), tokio_sqlite::Error> {
        let mut cache = self.cache.lock().await;
        let now = chrono::Utc::now().naive_utc();
        cache.retain(|_, (_, _, expiration, expiry)| {
            *expiration > now && *expiry > now
        });

        Ok(())
    }

    async fn store_authorization_in_cache(
        &self,
        token: String,
        uid: i32,
        path: String,
        expiration: chrono::NaiveDateTime,
    ) {
        let cache_expiry = expiration + chrono::Duration::minutes(5);
        let mut cache = self.cache.lock().await;
        cache.insert(token, (uid, path, expiration, cache_expiry));
    }

    async fn get_authorization_from_cache(
        &self,
        token: String,
    ) -> Option<(i32, String, chrono::NaiveDateTime)> {
        let mut cache = self.cache.lock().await;
        if let Some((uid, path, expiration, _)) = cache.get(&token) {
            let now = chrono::Utc::now().naive_utc();
            if *expiration > now {
                let (uid, path, expiration) = (*uid, path.clone(), *expiration);
                cache.insert(
                    token,
                    (uid, path.clone(), expiration, now + chrono::Duration::minutes(5)),
                );
                return Some((uid, path, expiration));
            } else {
                cache.remove(&token);
            }
        }
        None
    }

    pub async fn get_token(&self, token: String) -> Result<Option<(i32, String, chrono::NaiveDateTime)>, tokio_sqlite::Error> {
        if let Some((uid, path, expiration)) = self.get_authorization_from_cache(token.clone()).await {
            return Ok(Some((uid, path, expiration)));
        }

        let mut lock = self.connection.lock().await;
        let row = lock.query_row(
            "SELECT uid, path, expiration FROM authorizations WHERE token = ?1",
            vec![token.clone().into()],
        ).await?;

        if let Some(row) = row {
            let values = row.into_values();
            let uid = match values[0] {
                tokio_sqlite::Value::Integer(i) => i,
                _ => return Err(tokio_sqlite::Error::ExecuteReturnedResults),
            };
            let path = match &values[1] {
                tokio_sqlite::Value::Text(s) => s.clone(),
                _ => return Err(tokio_sqlite::Error::ExecuteReturnedResults),
            };
            let expiration_str = match &values[2] {
                tokio_sqlite::Value::Text(s) => s.clone(),
                _ => return Err(tokio_sqlite::Error::ExecuteReturnedResults),
            };
            let expiration = chrono::NaiveDateTime::parse_from_str(&expiration_str, "%Y-%m-%d %H:%M:%S")
                .map_err(|_| tokio_sqlite::Error::ExecuteReturnedResults)?;
            
            drop(lock);
            self.store_authorization_in_cache(token, uid as i32, path.clone(), expiration).await;
            
            return Ok(Some((uid as i32, path, expiration)));
        }

        Ok(None)
    }

    pub async fn create_token(
        &self,
        uid: i32,
        path: String,
    ) -> Result<(String, chrono::NaiveDateTime), tokio_sqlite::Error> {
        let token = uuid::Uuid::new_v4().to_string();
        let expiration = chrono::Utc::now().naive_utc() + chrono::Duration::hours(24);
        let expiration_str = expiration.format("%Y-%m-%d %H:%M:%S").to_string();

        let mut lock = self.connection.lock().await;
        lock.execute(
            "INSERT INTO authorizations (uid, path, token, expiration) VALUES (?1, ?2, ?3, ?4)",
            vec![
                uid.into(),
                path.clone().into(),
                token.clone().into(),
                expiration_str.into(),
            ],
        ).await?;

        self.store_authorization_in_cache(token.clone(), uid, path, expiration).await;
        Ok((token, expiration))
    }

    pub async fn check_if_path_and_uid_used_before(&self, uid: u32, path: String) -> Result<bool, tokio_sqlite::Error> {
        let mut lock = self.connection.lock().await;
        let result = lock.query_row(
            "SELECT EXISTS(SELECT 1 FROM authorizations WHERE uid = ?1 AND path = ?2)",
            vec![
                uid.into(),
                path.into(),
            ],
        ).await?;

        if let Some(row) = result {
            let values = row.into_values();
            if let tokio_sqlite::Value::Integer(exists) = values[0] {
                return Ok(exists != 0);
            }
        }
        Ok(false)
    }

    pub async fn renew_token_expiration(&self, token: String) -> Result<Option<chrono::NaiveDateTime>, tokio_sqlite::Error> {
        let now = chrono::Utc::now().naive_utc();
        let new_expiration = now + chrono::Duration::hours(24);
        let new_expiration_str = new_expiration.format("%Y-%m-%d %H:%M:%S").to_string();

        let mut lock = self.connection.lock().await;
        let res = lock.execute(
            "UPDATE authorizations SET expiration = ?1 WHERE token = ?2",
            vec![
                new_expiration_str.into(),
                token.clone().into(),
            ],
        ).await?;
        if res.rows_affected() == 0 {
            return Ok(None);
        }

        let mut cache = self.cache.lock().await;
        if let Some((uid, path, expiration, cache_expiration)) = cache.get_mut(&token) {
            *uid = *uid;
            *path = path.clone();
            *expiration = new_expiration;
            *cache_expiration = now + chrono::Duration::minutes(5);
        }

        Ok(Some(new_expiration))
    }
}
