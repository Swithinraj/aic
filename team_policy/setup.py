from setuptools import find_packages, setup

package_name = "team_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team",
    maintainer_email="team@example.com",
    description="Planner-first team policy package for the AI for Industry Challenge",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rviz_click_to_move = team_policy.planner.rviz_click_to_move:main",
        ],
    },
)
