import numpy as np
import time
import math
import os

class Twist:
    class Linear:
        x, y, z = 0.0, 0.0, 0.0
    linear = Linear()

class M:
    def _command_small_translation(self, delta_xyz: np.ndarray, max_step: float) -> float:
        # Move by delta_xyz vector. Returns actual distance moved.
        norm = float(np.linalg.norm(delta_xyz))
        if norm < 1e-7: return 0.0
        if norm > max_step: delta_xyz = delta_xyz / norm * max_step
        
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        d = delta_xyz
        dist = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        vx, vy, vz = d[0]/dist*speed, d[1]/dist*speed, d[2]/dist*speed
        
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = float(vz)
        
        cur = self._motion_servo.get_current_pose()
        if cur is None: return 0.0
        sx, sy, sz = float(cur.position.x), float(cur.position.y), float(cur.position.z)
        
        t0 = time.monotonic()
        timeout = dist / speed + 0.6
        while time.monotonic() - t0 < timeout:
            p = self._motion_servo.get_current_pose()
            if p is None: break
            moved = math.sqrt((p.position.x-sx)**2 + (p.position.y-sy)**2 + (p.position.z-sz)**2)
            if moved >= dist * 0.85: break
            self._motion_servo.publish_twist_command(twist, frame_id="base_link")
            self.sleep_for(0.02)
        self._motion_servo.stop()
        return float(moved)

    def _return_to_pose_xyz(self, target_pose, settle_sec: float = 0.20):
        # Similar to _return_to_pose_xy but 3D
        cur = self._motion_servo.get_current_pose()
        if cur is None or target_pose is None: return
        dx = float(target_pose.position.x - cur.position.x)
        dy = float(target_pose.position.y - cur.position.y)
        dz = float(target_pose.position.z - cur.position.z)
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 5e-5: return
        speed = float(self._motion_servo.max_linear_speed) * 0.20
        vx, vy, vz = dx/dist*speed, dy/dist*speed, dz/dist*speed
        sx, sy, sz = float(cur.position.x), float(cur.position.y), float(cur.position.z)
        t0 = time.monotonic()
        while time.monotonic() - t0 < settle_sec * 4.0:
            p = self._motion_servo.get_current_pose()
            if p is None: break
            moved = math.sqrt((p.position.x-sx)**2 + (p.position.y-sy)**2 + (p.position.z-sz)**2)
            if moved >= dist * 0.90: break
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.linear.z = float(vx), float(vy), float(vz)
            self._motion_servo.publish_twist_command(tw, frame_id="base_link")
            self.sleep_for(0.025)
        self._motion_servo.stop()
        self.sleep_for(settle_sec * 0.5)

    def _get_correction_plane_axes(self):
        pose = self._motion_servo.get_current_pose()
        if pose is None: return np.array([1.,0.,0.]), np.array([0.,1.,0.])
        
        w, x, y, z = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
        z_insert = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x*x + y*y)
        ])
        
        world_z = np.array([0., 0., 1.])
        a = np.cross(world_z, z_insert)
        if np.linalg.norm(a) < 1e-3:
            a = np.array([1., 0., 0.])
        else:
            a = a / np.linalg.norm(a)
            
        b = np.cross(z_insert, a)
        b = b / np.linalg.norm(b)
        return a, b
        
    def _check_visual_angle_alignment(self, camera, target_port_name, get_observation, send_feedback):
        return True

    def _bayesian_evidence_visual_servo_align(self, target_port_name, get_observation, send_feedback):
        pass
