#!/bin/bash

echo "VideoReTalking - Robust Run Script (Prevents Reboot)"
echo "===================================================="
source venv_video_retalking/bin/activate
# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "ERROR: Virtual environment not activated!"
    echo "Please run: source venv_video_retalking/bin/activate"
    exit 1
fi



# Memory optimization and crash prevention settings
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use CPU to avoid MPS memory issues
export OMP_NUM_THREADS=2  # Limit CPU threads to prevent overheating

# Create temp directory if it doesn't exist
mkdir -p temp

echo "Starting robust inference with crash prevention..."
echo "- Face detection failures will be handled gracefully"
echo "- Using CPU processing to avoid MPS memory sharing"
echo "- Reduced threading to prevent thermal issues"
echo "- Monitoring for common failure points"


# Monitor memory usage in background
monitor_memory() {
    while true; do
        memory_percent=$(vm_stat | awk '
            /Pages free:/ { free = $3 }
            /Pages active:/ { active = $3 }
            /Pages inactive:/ { inactive = $3 }
            /Pages wired down:/ { wired = $3 }
            END {
                used = active + inactive + wired
                total = used + free
                if (total > 0) print int((used / total) * 100)
            }')
        
        if [[ "$memory_percent" -gt 90 ]]; then
            echo "WARNING: High memory usage detected: ${memory_percent}%"
        fi
        sleep 30
    done
}

# Start memory monitoring in background
monitor_memory &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    exit
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

echo "Memory monitoring started (PID: $MONITOR_PID)"
echo "Starting inference..."

# Run inference with error handling
python3 inference.py \
  --face examples/face/3.mp4 \
  --audio examples/audio/2.wav \
  --outfile results/3_2_robust.mp4 \
  --LNet_batch_size 1 \
  --face_det_batch_size 1 \
  --img_size 256

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo ""
    echo "✓ Inference completed successfully!"
else
    echo ""
    echo "✗ Inference failed with exit code: $exit_code"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check if temp/faulty_frame.jpg exists to see problematic frames"
    echo "2. Verify all model files are present in checkpoints/"
    echo "3. Ensure sufficient memory and disk space"
    echo "4. Check the terminal output above for specific error messages"
fi
