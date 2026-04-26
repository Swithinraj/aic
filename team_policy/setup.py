from pathlib import Path
from setuptools import find_packages, setup

package_name = "team_policy"
package_root = Path(package_name) if Path(package_name).is_dir() else Path(".")


def package_files(*roots):
    files = []
    for root in roots:
        root_path = package_root / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file():
                files.append(str(path.relative_to(package_root)))
    return files


def share_files(root):
    entries = []
    root_path = package_root / root
    if not root_path.exists():
        return entries
    for path in root_path.rglob("*"):
        if path.is_file():
            rel_parent = path.relative_to(package_root).parent
            entries.append((str(Path("share") / package_name / rel_parent), [str(path)]))
    return entries


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *share_files("models"),
    ],
    include_package_data=True,
    package_data={
        "team_policy": package_files("models", "perception"),
    },
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Team",
    maintainer_email="team@example.com",
    description="Planner-first team policy package for the AI for Industry Challenge",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rviz_click_to_move = team_policy.planner.rviz_click_to_move:main",
            "test_stereo_center_depth = team_policy.Depth.test_depth_publisher:main",
            "test_depth_axis_estimator = team_policy.planner.depth_angle_estimator:main",
            "test_yolov12_perception = team_policy.perception.yolov12_detector:main",
            "test_pose_estimator = team_policy.perception.pose_estimator:main",
            "combined_yolo_depth_pose_planner = team_policy.planner.combined_yolo_depth_pose_planner:main",
        ],
    },
)