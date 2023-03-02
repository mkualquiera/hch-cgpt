use serde::{Deserialize, Serialize};

/// Roles that can be used in a chat log
#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum ChatRole {
    /// The system, used for the initial prompt and maybe other things
    #[serde(rename = "system")]
    System,
    /// The user, used for the user's input
    #[serde(rename = "user")]
    User,
    /// The assistant, used for the assistant's response
    #[serde(rename = "assistant")]
    Assistant,
}

/// A single entry in a chat log
#[derive(Serialize, Deserialize, Debug)]
struct ChatEntry {
    /// The role of the entry
    role: ChatRole,
    /// The text of the entry
    content: String,
}

/// A chat completion request
#[derive(Serialize, Deserialize, Debug)]
struct ChatCompletionRequest {
    /// The model used for the completion
    model: String,
    /// The chat log
    messages: ChatLog,
}

impl ChatCompletionRequest {
    /// Create a new chat completion request
    fn new(model: String, messages: ChatLog) -> ChatCompletionRequest {
        ChatCompletionRequest { model, messages }
    }
}

impl From<ChatLog> for ChatCompletionRequest {
    /// Create a new chat completion request from a chat log
    fn from(log: ChatLog) -> ChatCompletionRequest {
        ChatCompletionRequest::new("gpt-3.5-turbo".to_string(), log)
    }
}

/// A chat log, which is a list of chat entries
#[derive(Serialize, Deserialize, Debug)]
struct ChatLog(Vec<ChatEntry>);

/// A reason for which the completion stopped
#[derive(Serialize, Deserialize, Debug)]
enum FinishReason {
    /// A stop token was reached
    #[serde(rename = "stop")]
    Stop,
    /// The maximum number of tokens was reached
    #[serde(rename = "length")]
    Length,
}

/// Chat completion choice
#[derive(Serialize, Deserialize, Debug)]
struct ChatCompletionChoice {
    /// The text of the choice
    index: usize,
    /// The message of the choice
    message: ChatEntry,
    /// The finish reason of the choice
    finish_reason: FinishReason,
}

/// A completion usage information
#[derive(Serialize, Deserialize, Debug)]
struct CompletionUsage {
    /// The tokens in the prompt
    prompt_tokens: usize,
    /// The tokens in the completion
    completion_tokens: usize,
    /// The tokens in the total
    total_tokens: usize,
}

/// A chat completion response
#[derive(Serialize, Deserialize, Debug)]
struct ChatCompletionResponse {
    /// The completion id
    id: String,
    /// The completion object
    object: String,
    /// The completion creation time
    created: usize,
    /// The completion choices
    choices: Vec<ChatCompletionChoice>,
    /// The completion usage
    usage: CompletionUsage,
}

/// OpenAI api clients
pub struct OpenAI {
    /// HTTP client
    client: reqwest::blocking::Client,
    /// OpenAI api key
    api_key: String,
}

impl OpenAI {
    /// Create a new OpenAI client
    pub fn new(api_key: String) -> OpenAI {
        OpenAI {
            client: reqwest::blocking::Client::new(),
            api_key,
        }
    }

    /// Complete a chat
    fn complete_chat(
        &self,
        chat: ChatLog,
    ) -> Result<ChatCompletionResponse, reqwest::Error> {
        let request = ChatCompletionRequest::from(chat);

        // Make post request to OpenAI
        self.client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(self.api_key.clone())
            .json(&request)
            .send()?
            .json::<ChatCompletionResponse>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test serialization of a chat log
    #[test]
    fn test_chat_log() {
        let log = ChatLog(vec![
            ChatEntry {
                role: ChatRole::System,
                content: "Hello, world!".to_string(),
            },
            ChatEntry {
                role: ChatRole::User,
                content: "Hello, world!".to_string(),
            },
            ChatEntry {
                role: ChatRole::Assistant,
                content: "Hello, world!".to_string(),
            },
        ]);
        let serialized = serde_json::to_string(&log).unwrap();
        insta::assert_yaml_snapshot!(serialized);
    }

    use std::env;

    /// Test the chat completion request
    #[test]
    fn test_chat_completion_request() {
        // Read the key from environment variable
        let key = env::var("OPENAI_KEY").expect("OPENAI_KEY must be set");

        // Create a new OpenAI client
        let openai = OpenAI::new(key);

        // Create a new chat log
        let log = ChatLog(vec![
            ChatEntry {
                role: ChatRole::System,
                content: "You are an assistant that always says \"A\"".to_string(),
            },
            ChatEntry {
                role: ChatRole::User,
                content: "Please say \"A\". Do not say anything else, only \"A\"."
                    .to_string(),
            },
        ]);

        // Complete the chat
        let response = openai.complete_chat(log).expect("Failed to complete chat");

        // Get the first choice
        let choice = response.choices.first().expect("No choices");

        // Get the message
        let message = &choice.message;

        // Check that the message is correct
        assert_eq!(message.role, ChatRole::Assistant);

        println!("Assistant: {}", message.content);
    }
}
