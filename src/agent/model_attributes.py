# Define expected dimensions (replace with your actual values)
DEPTH_MAP_CHANNELS = 1
DEPTH_MAP_HEIGHT = 64
DEPTH_MAP_WIDTH = 64
MAX_OTHER_NODES = 10
QUATERNION_DIM = 4
POSITION_DIM = 3
VECTOR_INPUT_DIM = QUATERNION_DIM + POSITION_DIM  # Now 7
OTHER_NODE_FEATURE_DIM = QUATERNION_DIM + POSITION_DIM  # Now 7
EMBEDDING_DIM = 128  # Example dimension for processed features
HEADCOUNT = 3  # The amount of encoding heads
