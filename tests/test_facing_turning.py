"""Check that torso-facing and native OpenFace head rotation stay independent."""

import math
import unittest
import numpy as np

from behaviour_common import TemporalGate
from body_gestures import BodyGestureDetector
from facial_expressions import classify_observation
from main import load_settings


def speaker(face_x=.5, yaw=0):
    pose = np.full((33, 5), np.nan)
    pose[:11] = [[x, .15, 0, .99, .99] for x in np.linspace(face_x-.02, face_x+.02, 11)]
    for i, xy in {11: (.6,.3),12: (.4,.3),23: (.55,.6),24: (.45,.6)}.items():
        pose[i] = [*xy, 0, .99, .99]
    world = pose.copy()
    a = .15*np.array([np.cos(np.radians(yaw)),0,np.sin(np.radians(yaw))])
    world[11,:3], world[12,:3] = a,-a
    world[23,:3], world[24,:3] = a+[0,.4,0],-a+[0,.4,0]
    return pose, world


def head(yaw):
    return {"success": 1, "confidence": .99, "pose_Rx": 0, "pose_Ry": -math.radians(yaw), "pose_Rz": 0}


class FacingTurningTests(unittest.TestCase):
    def test_rear_camera_defaults(self):
        for yaw, facing_class, facing_board in ((0,True,False),(180,False,True),(90,False,False)):
            states,_=BodyGestureDetector(1000,1000).update(*speaker(yaw=yaw),0)
            self.assertIs(states["facing_class"],facing_class)
            self.assertIs(states["facing_board"],facing_board)
        _,_,scene,_=load_settings()
        self.assertEqual(scene,{"board_yaw_deg":180.,"class_yaw_deg":0.})

    def test_yaw_deadband_and_sign(self):
        for yaw,left,right in ((-20,True,False),(-15,False,False),(0,False,False),(15,False,False),(20,False,True)):
            states,info=classify_observation(head(yaw),{})
            self.assertIs(states["turning_left"],left)
            self.assertIs(states["turning_right"],right)
            self.assertAlmostEqual(info["yaw_image_deg"],yaw)

    def test_sideways_lean_cannot_create_head_turn_from_body(self):
        for x in (.3,.5,.7):
            body,_=BodyGestureDetector(1000,1000).update(*speaker(face_x=x),0)
            self.assertNotIn("turning",body)
            face,_=classify_observation(head(0),{})
            self.assertFalse(face["turning"])

    def test_head_does_not_require_shoulders_and_needs_temporal_confirmation(self):
        gate=TemporalGate(.25)
        for t in (0,.1,.2,.3):
            raw,_=classify_observation(head(-20),{})
            states=gate.update(raw,t)
            self.assertIs(states["turning"],t>=.25)
        raw,_=classify_observation({"success":0,"confidence":1}, {})
        self.assertIsNone(gate.update(raw,.4)["turning"])


if __name__ == "__main__":
    unittest.main()
