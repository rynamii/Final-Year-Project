#!/bin/bash
# To run use `source path_setup.sh`

# Add directories to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:pose_finder/utils/mano_core/"
export PYTHONPATH="${PYTHONPATH}:pose_finder/"
export PYTHONPATH="${PYTHONPATH}:points_detector/"
export PYTHONPATH="${PYTHONPATH}:hand_detection/"

# Optional: Display the updated PYTHONPATH
echo "Updated PYTHONPATH:"
echo $PYTHONPATH