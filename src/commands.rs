use crossbeam_channel::Sender;

use crate::types::*;

#[derive(Debug)]
pub enum EngineCommands {
    AddNode { 
        position: Position,
        rotation: Rotation,
        response_tx: Option<Sender<EngineResponses>> 
    },
    GetFullGraphState {
        response_tx: Option<Sender<EngineResponses>> 
    },
}

#[derive(Debug)]
pub enum EngineResponses {
    AddNodeResponse { id: NodeId },
    FullGraphStateResponse { full_graph_state: FullGraphState },
}
