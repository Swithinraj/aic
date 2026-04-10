from glob import glob
from setuptools import find_packages, setup

package_name = "team_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/models", glob("models/*.pt")),
    ],
    include_package_data=True,
    package_data={
        'team_policy': [
            'models/*.pt',
            'perception/*.png',
            'perception/*.json',
        ],
    },
    install_requires=["setuptools"],
    zip_safe=True,
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
