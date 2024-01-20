#!/usr/bin/env bash
set -Eeuo pipefail
# Check if APP_CONTEXT matches one of the specific values
if [ "$APP_CONTEXT" = "1024" ]; then
    echo "APP_CONTEXT is 1024"
    /usr/bin/python /workspace/app/app.py "$@"
elif [ "$APP_CONTEXT" = "512" ]; then
    echo "APP_CONTEXT is 512"
    /usr/bin/python /workspace/app/app_512.py "$@"
elif [ "$APP_CONTEXT" = "LCM" ]; then
    echo "APP_CONTEXT is LCM"
    /usr/bin/python /workspace/app/app_lcm.py "$@"
else
    echo "APP_CONTEXT is not set to 1024, 512, or LCM, defaulting to 1024"
    /usr/bin/python /workspace/app/app.py "$@"
fi

