#!/usr/bin/env bash
# 量測效能前固定電源模式與時脈，並印出模式（報告需註明）
# 模式編號依機型 / JetPack 而不同（Orin Nano 8GB：0=15W、1=25W、2=MAXN_SUPER），所以預設用「名稱」從 /etc/nvpmodel.conf 查編號，不寫死。
#   ./05_set_power_mode.sh            # 名稱以 MAXN 開頭的模式（此機為 MAXN_SUPER）
#   ./05_set_power_mode.sh 15W        # 以名稱選模式
#   ./05_set_power_mode.sh 1          # 純數字視為編號；對照表見 `grep POWER_MODEL /etc/nvpmodel.conf`
set -euo pipefail
CONF=${NVPMODEL_CONF:-/etc/nvpmodel.conf}
want=${1:-MAXN}
if [[ $want =~ ^[0-9]+$ ]]; then
    id=$want
else
    id=$(grep -oE "POWER_MODEL ID=[0-9]+ NAME=${want}[A-Za-z0-9_]*" "$CONF" | head -1 | sed -E 's/.*ID=([0-9]+).*/\1/' || true)
    [ -n "$id" ] || { echo "在 $CONF 找不到名稱以 '$want' 開頭的電源模式。可用模式："; grep -oE "ID=[0-9]+ NAME=[A-Za-z0-9_]+" "$CONF"; exit 1; }
fi
echo "切換到 nvpmodel 模式 $id（$(grep -oE "ID=$id NAME=[A-Za-z0-9_]+" "$CONF" | sed 's/.*NAME=//')）"
sudo nvpmodel -m "$id"
sudo jetson_clocks
sudo nvpmodel -q
