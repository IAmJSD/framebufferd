use std::sync::mpsc::{channel, Sender};
use tokio::sync::oneshot;

pub struct PrompterRequest {
    // The message to display to the user
    pub message: String,

    // The channel to send the user's reply back
    pub reply_channel: oneshot::Sender<bool>,
}

// Creates a new thread that prompts the user for input and sends it to the provided channel.
pub fn start_prompter() -> Sender<PrompterRequest> {
    let (tx, rx) = channel::<PrompterRequest>();

    std::thread::spawn(move || {
        // Handle incoming requests
        for request in rx {
            // Display a GUI dialog with Yes/No buttons
            let response = rfd::MessageDialog::new()
                .set_title("Permission Request")
                .set_description(&request.message)
                .set_buttons(rfd::MessageButtons::YesNo)
                .show();

            // Convert the response to a boolean
            let answer = matches!(response, rfd::MessageDialogResult::Yes);

            // Send the response back through the reply channel
            let _ = request.reply_channel.send(answer);
        }
    });

    tx
}
