use cogfer::transport::mock::MockTransport;
use cogfer::{ErrorKind, FinishReason, Message, ReasoningConfig, Request, StreamEvent};
use serde_json::json;

use crate::common::*;

mod request;
mod response;
mod streaming;
