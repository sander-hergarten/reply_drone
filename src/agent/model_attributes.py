# agent/model_attributes.py

# Define expected dimensions for raw inputs
DEPTH_MAP_CHANNELS = 1
DEPTH_MAP_HEIGHT = 64
DEPTH_MAP_WIDTH = 64
MAX_OTHER_NODES = 10  # Max sequence length for other nodes Transformer

# Base dimensions
POSITION_DIM = 3  # X, Y, Z position
ROTATION_DIM = 1  # Represents the Z-axis rotation angle

# --- MODIFIED: Define output dimension for each processor head ---
# Each individual processor (vector, cnn, other_nodes, etc.)
# will output an embedding of this size.
PARTIAL_EMBEDDING_DIM = 64  # Example value, adjust as needed
# --- END MODIFIED ---

# --- REMOVED: HEADCOUNT is no longer needed, determined by number of processors ---
# HEADCOUNT = 3
# --- END REMOVED ---

# --- MODIFIED: Feature dimensions based on raw inputs ---
# Dimension for the agent's own vector state (Position + Rotation Angle)
VECTOR_INPUT_DIM = POSITION_DIM + ROTATION_DIM
# Feature dimension for other nodes (Position + Rotation Angle) before processing
OTHER_NODE_RAW_FEATURE_DIM = POSITION_DIM + ROTATION_DIM
# --- END MODIFIED ---

# --- KEPT FOR REFERENCE (Used in CNN calculation) ---
# EMBEDDING_DIM = 128 # Old combined dimension, replace usage with dynamic calculation
# --- END KEPT ---

# --- ADDED: Dimension for Transformer internal processing ---
# This should ideally be related to PARTIAL_EMBEDDING_DIM or a multiple
# for consistency, but can be independent. Let's make it a multiple for now.
TRANSFORMER_D_MODEL = PARTIAL_EMBEDDING_DIM * 2  # Example: 128
# --- END ADDED ---
