"""Exercise source-time joins, native quality masks and physical-motion proxies."""

import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_movement import CameraGuard, CameraMovementDetector
from facial_expressions import (AU_NUMBERS, CycleDetector, FacialExpressionExtractor, HMADTables,
                                IntervalTable, build_face_timeline, classify_observation)
from main import main
from native_tracking import sha256, validate_openface_csv
from tracking_config import GazeRules, TrackingRules
from teaching_aids import TeachingAidTracker, TeachingAidRules


def observation(t=0, **overrides):
    row={"frame":round(t*25)+1,"timestamp":t,"success":1,"confidence":.99,
         "pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.,"pose_Tz":1500.,"gaze_angle_x":0.,"gaze_angle_y":0.}
    row.update({f"AU{i:02d}_r":0. for i in AU_NUMBERS})
    row.update({f"gaze_{eye}_{axis}":(-1. if axis=="z" else 0.) for eye in (0,1) for axis in "xyz"})
    row.update({f"{axis}_{i}":32. for i in range(68) for axis in "xy"})
    row.update(overrides)
    return row


def exports(root, rows, overrides=None):
    csv_path=root/"openface.csv"
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    directory=root/"hmad"; directory.mkdir()
    (directory/"status.tsv").write_text('component\tstatus\treason\nsmile\tok\t\nbrows\tok\t\nbagmad\tok\t\n')
    end=len(rows)/25
    tables={"smile":[(0,end,'S2')],"brow_raising":[(0,end,'A')],"brow_lowering":[(0,end,'N')],
            "blinking":[(0,end,'no blink')],"gaze":[(0,end,'eye contact')]}
    tables.update(overrides or {})
    for name, intervals in tables.items():
        with (directory/f'{name}.tsv').open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t'); w.writerow(['start_s','end_s','value']); w.writerows(intervals)
    return csv_path,directory


def torso(scale=1):
    p=np.full((33,5),np.nan)
    for i,xy in {11:(.4,.3),12:(.6,.3),23:(.43,.6),24:(.57,.6)}.items():
        xy=(np.asarray(xy)-.5)*scale+.5
        p[i]=[*xy,0,.99,.99]
    return p


class TrackingTests(unittest.TestCase):
    def test_head_gaze_quality_and_blink_are_independent(self):
        native={'smile':'S4','brow_raising':'N','brow_lowering':'B','blinking':'blink','gaze':'eye contact'}
        s,i=classify_observation(observation(),native)
        self.assertTrue(s['smiling']);self.assertTrue(s['laughing_smile']);self.assertTrue(s['blinking'])
        self.assertIsNone(s['gaze_toward_camera']);self.assertTrue(i['head_valid'])
        native['blinking']='no blink'
        s,i=classify_observation(observation(gaze_angle_x=math.radians(20)),native)
        self.assertTrue(s['gaze_right']);self.assertIsNone(s['gaze_toward_reference'])
        rules=replace(TrackingRules(),gaze=GazeRules(reference_validated=True,reference_description='annotated partner region'))
        self.assertTrue(classify_observation(observation(),native,rules)[0]['gaze_toward_reference'])
        self.assertTrue(all(v is None for v in classify_observation(observation(confidence=.8),native)[0].values()))

    def test_x_and_missing_are_unknown_not_neutral(self):
        s,i=classify_observation(observation(),{'smile':'X','brow_raising':'X','brow_lowering':'N'})
        self.assertIsNone(s['smiling']);self.assertIsNone(i['smile_level']);self.assertFalse(s['brow_lowering'])

    def test_cycles_once_monotonic_and_loss(self):
        d=CycleDetector('head_nodding')
        events=[e for n,v in enumerate([0,2,5,10,7,3,0,0]) if (e:=d.update(n*.1,v))]
        self.assertEqual(len(events),1)
        self.assertGreaterEqual(events[0]['excursion_deg'],8)
        d=CycleDetector('head_shaking')
        self.assertFalse(any(d.update(n*.1,n*2) for n in range(15)))
        d.reset();d.update(0,0);d.update(.1,10);d.update(.2,None)
        self.assertIsNone(d.update(.3,0))

    def test_native_intervals_preserve_brief_blinks_and_invalid_tracking(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            rows=[observation(i/25,success=0 if i==4 else 1) for i in range(10)]
            path,hmad=exports(root,rows,{'blinking':[(0,.08,'no blink'),(.08,.12,'blink'),(.12,.4,'no blink')]})
            assets=build_face_timeline(path,hmad,root/'face.jsonl')
            records=[json.loads(line) for line in assets['timeline'].read_text().splitlines()]
            self.assertTrue(records[2]['states']['blinking'])
            self.assertIsNone(records[4]['states']['smiling'])
            events=[json.loads(line) for line in assets['annotations'].read_text().splitlines()]
            self.assertEqual([(e['start_s'],e['end_s']) for e in events if e['behaviour']=='blinking'],[(.08,.12)])
            with FacialExpressionExtractor(assets['timeline'],assets['cycles']) as reader:
                s,info,_=reader.process(.1001,.1335)
                self.assertTrue(s['blinking']);self.assertAlmostEqual(info['analysis_timestamp_s'],.08)
                self.assertIsNone(reader.process(.1701,.2)[0]['smiling'])
                self.assertIsNone(reader.process(.5,.533)[0]['smiling'])

    def test_csv_rejects_wrong_fps_and_overlapping_intervals(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path,_=exports(root,[observation(0),observation(.04)])
            self.assertEqual(validate_openface_csv(path,.08),2)
            rows=[observation(0),observation(1/30)]
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            with self.assertRaises(ValueError):validate_openface_csv(path,.08)
            table=root/'bad.tsv';table.write_text('start_s\tend_s\tvalue\n0\t1\tS0\n0.5\t2\tS1\n')
            with self.assertRaises(ValueError):IntervalTable(table,('S0','S1'))

    def test_zero_width_native_boundaries_are_counted_without_inventing_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            table=Path(name)/'native.tsv'
            table.write_text('start_s\tend_s\tvalue\n0\t0.1\tN\n0.1\t0.1\tX\n0.1\t0.2\tA\n')
            values=IntervalTable(table,('N','X','A'))
            self.assertEqual(values.zero_duration_rows,1)
            self.assertEqual(values.value(.1),'A')
            self.assertIsNone(values.value(.2))

    def test_blink_and_coarse_gaze_survive_failed_reference_fit(self):
        with tempfile.TemporaryDirectory() as name:
            path,hmad=exports(Path(name),[observation()])
            (hmad/'status.tsv').write_text('component\tstatus\treason\nsmile\tok\t\nbrows\tok\t\n'
                'bagmad\tpartial\t\nbagmad_blinks\tok\t\nbagmad_gaze\terror\tdegenerate reference\n')
            native=HMADTables(hmad).sample(0)
            states,info=classify_observation(observation(),native)
            self.assertFalse(states['blinking'])
            self.assertTrue(states['gaze_toward_camera'])
            self.assertIsNone(states['gaze_toward_reference'])

    def test_import_may_use_a_verified_longer_prefix_but_not_shorter_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            path,_=exports(Path(name),[observation(i/25) for i in range(25)])
            self.assertEqual(validate_openface_csv(path,.4,allow_longer=True),25)
            with self.assertRaises(ValueError):validate_openface_csv(path,1.2,allow_longer=True)

    def test_camera_motion_requires_body_correspondence_and_stability(self):
        detector=CameraMovementDetector()
        for n in range(16):
            t=n/10
            s,i=detector.update(t,1500*(1-.06*t),torso(1+.06*t),1000,1000,0,True)
        self.assertTrue(s['moving_toward_camera'])
        detector=CameraMovementDetector()
        for n in range(16):
            t=n/10
            s,i=detector.update(t,1500*(1+.06*t),torso(1-.06*t),1000,1000,0,True)
        self.assertTrue(s['moving_away_from_camera'])
        detector=CameraMovementDetector()
        for n in range(16):
            t=n/10
            s,i=detector.update(t,1500*(1-.06*t),torso(),1000,1000,0,True)
        self.assertIsNone(s['moving_toward_camera'])
        self.assertIsNone(detector.update(1.6,1400,torso(),1000,1000,0,False)[0]['moving_toward_camera'])

    def test_background_guard_marks_zoom_and_cuts(self):
        rng=np.random.default_rng(1)
        image=rng.integers(0,256,(180,320,3),dtype=np.uint8)
        guard=CameraGuard();guard.update(image,0)
        self.assertTrue(guard.update(image,.04)['camera_stable'])
        zoom=cv2.warpAffine(image,np.array([[1.08,0,-12],[0,1.08,-7]],dtype=float),(320,180))
        self.assertFalse(guard.update(zoom,.08)['camera_stable'])
        guard=CameraGuard();guard.update(np.zeros_like(image),0)
        self.assertTrue(guard.update(np.full_like(image,255),.04)['camera_cut'])

    def test_imported_native_exports_merge_into_real_video_outputs(self):
        from body_gestures import BODY_LABELS
        class FakeBody:
            def __init__(self,*a,**k):self.points={};self.hand_observations={}
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def process(self,*a):return dict.fromkeys(BODY_LABELS),{'pose_detected':False},None,np.full((33,5),np.nan),None
            def draw(self,*a):pass
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),29.97,(64,64))
            for i in range(12):writer.write(np.full((64,64,3),100,dtype=np.uint8))
            writer.release()
            path,hmad=exports(root,[observation(i/25) for i in range(10)])
            (hmad/'manifest.json').write_text(json.dumps({'openface_csv_sha256':sha256(path),'source_sha256':sha256(source),
                'roi_px':[0,0,64,64],'analysis_fps':25.,'hmad_revision':'test-fixture'}))
            with patch('main.BodyGestureExtractor',FakeBody):
                metadata=main(['--input',str(source),'--full-video','--openface-csv',str(path),
                               '--hmad-results',str(hmad),'--no-objects','--output-dir',str(root/'out')])
            self.assertEqual(metadata['processed_frames'],12)
            with Path(metadata['outputs']['frames']).open() as f:frames=list(csv.DictReader(f))
            self.assertEqual(len(frames),12);self.assertEqual(frames[0]['smiling'],'1')
            with Path(metadata['outputs']['events']).open() as f:events=list(csv.DictReader(f))
            self.assertTrue(any(e['behaviour']=='smiling' and e['timing_type']=='hmad_interval' for e in events))


class HoldingTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.full((33,5),np.nan)
        for i,xy in {11:(.4,.3),12:(.6,.3),15:(.35,.5),16:(.65,.5)}.items():self.pose[i]=[*xy,0,.99,.99]
        self.hand=np.tile([.35,.43,0.],(21,1))
        self.hand[0]=[.35,.5,0];self.hand[5]=[.32,.44,0];self.hand[17]=[.39,.44,0]
        self.hand[4]=[.351,.43,0];self.hand[8]=[.35,.43,0]
        self.observations={'hands':np.array([self.hand]),'hand_detected':[True],'handedness_scores':[.99]}
        self.obj={'class':'pen','score':.9,'box':[342.,418.,358.,445.]}
        self.frame=np.zeros((1000,1000,3),dtype=np.uint8)

    def test_holding_requires_sustained_contact_and_breaks_on_missing_hand(self):
        obj=self.obj
        class Detector:
            def detect(self,*a):return [obj]
        tracker=TeachingAidTracker(Detector())
        for n in range(6):
            raw,states,info=tracker.process(self.frame,n*.1,self.pose,self.observations)
        self.assertTrue(states['holding_pen']);self.assertEqual(info['holding_status'],'holding')
        self.assertEqual(info['held_objects'][0]['hand'],'left')
        _,states,info=tracker.process(self.frame,.6,self.pose,{})
        self.assertIsNone(states['holding_pen']);self.assertEqual(info['holding_status'],'unknown')

    def test_proximity_alone_and_ambiguous_types_do_not_confirm_holding(self):
        objects=[{**self.obj,'box':[500.,500.,520.,530.]}]
        class Detector:
            def detect(self,*a):return objects
        tracker=TeachingAidTracker(Detector())
        for n in range(6):raw,s,info=tracker.process(self.frame,n*.1,self.pose,self.observations)
        self.assertFalse(s['left_holding_object'])
        objects[:]=[self.obj,{**self.obj,'class':'marker','score':.89}]
        raw,s,info=tracker.process(self.frame,.6,self.pose,self.observations)
        self.assertIsNone(s['left_holding_object'])

    def test_object_relative_motion_breaks_confirmation(self):
        obj=self.obj.copy()
        class Detector:
            def detect(self,*a):return [obj]
        tracker=TeachingAidTracker(Detector())
        for n in range(6):raw,s,info=tracker.process(self.frame,n*.1,self.pose,self.observations)
        self.assertTrue(s['left_holding_object'])
        obj['box']=[300.,400.,400.,460.] # Nearby but switches object extent/center.
        obj['class']='book'
        raw,s,info=tracker.process(self.frame,.6,self.pose,self.observations)
        self.assertFalse(s['left_holding_object']);self.assertFalse(s['holding_book'])


if __name__=='__main__':unittest.main()
