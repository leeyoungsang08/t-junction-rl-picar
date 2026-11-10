#!/usr/bin/env bash
# One-shot setup for the PiCar Gazebo check on Ubuntu 22.04 (ROS 2 Humble + Gazebo Fortress).
#   bash picar_gazebo/scripts/setup_ubuntu2204.sh            # run from the repo root, asks for sudo
# Idempotent: safe to run again. Installs: ROS 2 Humble (desktop), ros-humble-ros-gz (Fortress),
# colcon, python deps; builds ~/picar_ws with picar_gazebo; adds PICAR_REPO + sourcing to ~/.bashrc.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${PICAR_WS:-$HOME/picar_ws}"
. /etc/os-release
if [ "${VERSION_ID}" != "22.04" ]; then
  echo "This script targets Ubuntu 22.04 (found ${PRETTY_NAME}). Humble/Fortress binaries need jammy." >&2
  exit 1
fi
echo "== repo: $REPO   workspace: $WS"

echo "== 1/6 locale + universe"
sudo apt-get update
sudo apt-get install -y locales software-properties-common curl git
sudo locale-gen en_US en_US.UTF-8 >/dev/null
sudo add-apt-repository -y universe

echo "== 2/6 ROS 2 apt source (ros2-apt-source package, current official method)"
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F "tag_name" | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME:-${VERSION_CODENAME}}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi
sudo apt-get update
sudo apt-get upgrade -y

echo "== 3/6 ROS 2 Humble + ros_gz (Gazebo Fortress) + tools"
sudo apt-get install -y ros-humble-desktop ros-humble-ros-gz ros-dev-tools python3-colcon-common-extensions \
  python3-numpy python3-yaml python3-matplotlib python3-pil python3-pytest ffmpeg

# shellcheck disable=SC1091
set +u; source /opt/ros/humble/setup.bash; set -u
if ! ign gazebo --versions >/dev/null 2>&1; then
  echo "== 'ign gazebo' CLI missing: adding packages.osrfoundation.org for ignition-fortress"
  sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable ${UBUNTU_CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y ignition-fortress
fi
ign gazebo --versions

echo "== 4/6 colcon workspace"
mkdir -p "$WS/src"
ln -sfn "$REPO/picar_gazebo" "$WS/src/picar_gazebo"
cd "$WS"
colcon build --symlink-install --packages-select picar_gazebo

echo "== 5/6 ~/.bashrc"
add_line() { grep -qxF "$1" "$HOME/.bashrc" || echo "$1" >> "$HOME/.bashrc"; }
add_line "source /opt/ros/humble/setup.bash"
add_line "source $WS/install/setup.bash"
add_line "export PICAR_REPO=$REPO"

echo "== 6/6 self-test (no Gazebo window): ROS-free checks + 2D kinematic pre-check"
cd "$REPO"
export PICAR_REPO="$REPO"
PYTHONPATH="$REPO:$REPO/picar_gazebo" python3 -m pytest -q picar_gazebo/test picar_lab/car_agent/tests
PYTHONPATH="$REPO:$REPO/picar_gazebo" python3 -m picar_gazebo.kinematic_check --cars 3 --seconds 60 --out /tmp/picar_kin \
  | python3 -c "import json,sys; m=json.load(sys.stdin); print({k: m[k] for k in ('crossings','collisions','faults','lane_departures','illegal_crossings')})"

cat <<MSG

Done. Open a NEW terminal (or: source ~/.bashrc), then
  ros2 launch picar_gazebo calibrate.launch.py          # ~25 s, writes ~/picar_gz_runs/car_calibration.yaml
  ros2 launch picar_gazebo course.launch.py cars:=3      # Gazebo window + 3 cars + central server
Results: ~/picar_gz_runs/<time>/metrics.json   (docs: docs/PICAR_GAZEBO_KO.md)
MSG
