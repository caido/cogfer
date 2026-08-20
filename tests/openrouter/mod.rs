use caido_ai::transport::mock::MockTransport;
use caido_ai::{ErrorKind, FinishReason, Message, ReasoningConfig, Request, StreamEvent};
use serde_json::json;

use crate::common::*;

mod request;
mod response;
mod streaming;
