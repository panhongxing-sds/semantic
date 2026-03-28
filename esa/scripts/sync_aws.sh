#!/bin/bash

# syncing output folder to S3
periodic_sync() {
  while true; do
    echo "Syncing local output folder to S3 bucket..."
    aws s3 sync $OUTPUT_PATH s3://$OUTPUT_BUCKET/$OUTPUT_FOLDER
    sleep 60  # Sync every 1 minute
  done
}

# Start the periodic sync in the background
periodic_sync &

# Capture the background process PID to stop it later
SYNC_PID=$!

# Ensure the background sync process is terminated when the script exits
trap "echo 'Stopping sync process'; kill $SYNC_PID" EXIT

python $TRAIN_SCRIPT

echo "Performing final sync to s3://$OUTPUT_BUCKET/$OUTPUT_FOLDER"
aws s3 sync $OUTPUT_PATH s3://$OUTPUT_BUCKET/$OUTPUT_FOLDER

echo "Training complete."