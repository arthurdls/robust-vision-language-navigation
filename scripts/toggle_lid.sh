#!/bin/bash

CONF="/etc/systemd/logind.conf"

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)."
    exit 1
fi

if [[ ! -f "$CONF" ]]; then
    echo "Config file $CONF not found."
    exit 1
fi

cp "$CONF" "${CONF}.bak"

set_option() {
    local key="$1"
    local value="$2"
    if grep -qE "^${key}=" "$CONF"; then
        sed -i "s/^${key}=.*/${key}=${value}/" "$CONF"
    elif grep -qE "^#${key}=" "$CONF"; then
        sed -i "s/^#${key}=.*/${key}=${value}/" "$CONF"
    else
        echo "${key}=${value}" >> "$CONF"
    fi
}

current=$(grep -E "^HandleLidSwitch=" "$CONF" 2>/dev/null | head -1 | cut -d= -f2)

if [[ "$current" == "ignore" ]]; then
    set_option "HandleLidSwitch" "suspend"
    set_option "HandleLidSwitchExternalPower" "suspend"
    new_state="ON"
else
    set_option "HandleLidSwitch" "ignore"
    set_option "HandleLidSwitchExternalPower" "ignore"
    new_state="OFF"
fi

systemctl kill -s HUP systemd-logind

echo "Suspend on lid close: $new_state"
echo "(Backup saved to ${CONF}.bak)"
