# Predicting a Penalty Kick Before the Foot Meets the Ball

## The problem

A penalty taker commits to a side a fraction of a second before contact, and the goalkeeper has to guess at the
same moment. Given a short broadcast video clip that **stops 0.02 s before the kicker's foot touches the ball**,
the task is to predict three things about what happens next:

| Target | Classes | Weight in the score |
|---|---|---|
| `kick_direction` | left, middle, right | 0.60 |
| `goalkeeper_direction` | left, stands in the middle, right | 0.25 |
| `goal_outcome` | goal, no_goal | 0.15 |

Left and right are from the kicker's point of view. The score is the weighted sum of the three accuracies.

The data are 2,200 pre-contact clips cut from 215 source videos (shootouts, match highlights and penalty
compilations, men's and women's football): 1,300 labelled clips for training and 900 unlabelled clips to predict.
Clips run at 25 to 60 fps, 640x360 to 1280x720, and the players are only 30 to 80 pixels tall. Almost every
source video sits entirely on one side of the split, so the model is evaluated on stadiums, broadcasters and
goalkeepers it has never seen. Each clip also comes with the kicker's preferred foot and the competition's gender.

Guessing the majority class scores 0.517; an "educated guess" from the preferred foot scores 0.550.

## The approach

Nothing after contact is visible, so the model reads **body posture in the last moments of the run-up**:
hip and shoulder orientation, where the plant foot points, how the kicking leg swings, where the goalkeeper has
already started to move.

1. **Find the two players.** A YOLO11x person detector runs on time-sampled frames of every clip. A small
   learned picker (trained on hand-clicked positions of the kicker and goalkeeper in the training clips) chooses
   which detection is the kicker and which is the goalkeeper. The kicker-to-goalkeeper direction in the image
   defines a per-clip *kicker frame*, so that "forward" and "lateral" mean the same thing in every camera view.
2. **Estimate keypoints on up-scaled crops.** Full-frame pose estimation misses players this small, so the pose
   models run on 2 to 3x enlarged crops of each player over the last 0.4 s: RTMPose-m and RTMPose-x with the
   Halpe-26 skeleton (body plus big toe, small toe and heel of each foot) and YOLO11m-pose. The RTMPose runs are
   repeated on mirrored crops (test-time augmentation) and the keypoint tables are merged by confidence.
3. **Turn keypoints into kinematic features.** For a few fixed times before contact: the orientation of the plant
   foot and the kicking foot (toe minus heel), shank and thigh directions, hip and shoulder lines, trunk lean,
   gaze, arm positions, joint velocities, the goalkeeper's lateral displacement and velocity, and dense optical
   flow inside the player boxes. Every feature is tagged by how it behaves under a horizontal mirror, which lets
   training double the data with mirrored copies (left and right labels swapped) and average predictions over the
   original and mirrored views at inference.
4. **Train kick heads on the kicker only, and only on the last 0.04 s.** The two decisions that mattered most.
   A kick head that also sees the goalkeeper's motion learns to copy the goalkeeper, who guesses the side right
   only 56 percent of the time; and the earlier time steps of the run-up are noise for the kick. Restricting
   the LightGBM head to kicker-only features from the final 0.04 s raised its kick accuracy from 0.636 to 0.664
   in every cross-validation fold. LightGBM, XGBoost and ExtraTrees heads are trained on this table for each
   keypoint source and averaged, together with heads on the raw keypoint coordinates of the last two time steps.
5. **Goalkeeper heads.** LightGBM and ExtraTrees on goalkeeper-only features, a small temporal CNN on the raw
   keypoint tensor, and VideoMAE fine-tuned on crops of the goalkeeper.
6. **Stack.** One multinomial logistic regression per target combines the heads' out-of-fold probabilities,
   with cross-validation grouped by source video throughout. The goalkeeper target uses goalkeeper heads only.
   `goal_outcome` is predicted as the constant `goal`: no learned head beat the 72 percent base rate.

## Results

Cross-validation on the 1,300 labelled clips, grouped by source video so that no stadium or goalkeeper is shared
between training and validation folds:

| | kick | goalkeeper | goal | weighted score |
|---|---|---|---|---|
| Final model | 0.668 | 0.918 | 0.722 | **0.739** |
| Educated guess (preferred-foot majority) | | | | 0.550 |

On the hidden test clips the final model scores about 0.76. The goalkeeper is easy to read once their dive has
started; the kick direction, worth 60 percent of the score, is the hard part: even the goalkeepers in these clips,
reacting live to the same cues, go the right way only 56 percent of the time.

## What did not help

Measured with the same grouped cross-validation and left out of the final model: end-to-end video models and crop
CNNs on the kicker (they predict the class prior with 1,300 clips), 3D pose (the depth estimate cannot resolve the
body's yaw), ball detection and ball geometry, aligning clips on the kicking foot's speed peak instead of the clip
end, per-source feature centering, self-training on the unlabelled clips, a probability threshold for the rare
`middle` class, forcing the metadata's preferred foot as the kicking foot, and hyper-parameter sweeps.

## Repository

Everything is in one notebook, `penalty_kick_pipeline.ipynb`:

- the pipeline modules (46 Python files) embedded as collapsed `%%writefile` cells near the top; running them
  writes the code to a `src/` folder next to the notebook,
- then every stage in order, from raw clips to the submission CSV, each with a short explanation of what it does,
  which pretrained model it uses, how long it takes and what it writes,
- the stack, the CSV writer, the results table and the list of pretrained models.

The notebook installs its dependencies in its first code cell and expects the clips and metadata under one data
root (`CV3_ROOT`, default `D:\cv3`, layout described in its setup section); caches, run outputs and the final CSV
are written there. Pretrained models used: YOLO11x and YOLO11m-pose (Ultralytics), RTMPose-m and RTMPose-x
Halpe-26 and RTMW3D-x (OpenMMLab, run through `rtmlib`), VideoMAE (MCG-NJU checkpoints via Hugging Face). No
external football data.
