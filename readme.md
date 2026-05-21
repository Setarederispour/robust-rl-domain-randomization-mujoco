# Starting code for final course project extension of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1XWE2NB-keFvF-EDT_5muoY8gdtZYwJIeo48IMsnr9l0/edit?usp=sharing).


## Training and Testing

This project supports the “project extension” grid:

- **2 environments:** `hopper`, `walker2d`
- **3 settings:** `no DR` (no domain randomization), `UDR` (uniform domain randomization), `ADR` (adaptive domain randomization)

The main scripts are:

- `train.py` — trains a PPO policy and saves model + logs in a clean folder structure
- `test.py` — loads a saved PPO policy and runs one episode on **source** and **target**, optionally rendering

### Output structure

`train.py` automatically organizes outputs as:

- `models/<env_family>/<domain>/<setting>/ppo_<run_id>.zip`
- `logs/<env_family>/<domain>/<setting>/<run_id>/`

Where:
- `<env_family>` is `hopper` or `walker2d`
- `<domain>` is the training domain: `source` or `target`
- `<setting>` is one of:
  - `nodr`
  - `udr_r0.20` (ratio value included)
  - `adr` (only when ADR is implemented)
- `<run_id>` is either:
  - your custom `--run_name`, or
  - an auto-generated id like: `lr0.0003_ns2048_nenv8_seed0_ts1000000`

Stable-Baselines3 adds the `.zip` extension automatically.

### Train (Hopper)

No DR (baseline):
```bash
python train.py --env_family hopper --domain source --no-udr
```

UDR (default ratio 0.20):

```bash
python train.py --env_family hopper --domain source --udr --udr_ratio 0.20
```

Same, but with a custom name (simplifies file paths later):
```bash
python train.py --env_family hopper --domain source --udr --udr_ratio 0.20 --run_name hopper_udr_seed0
```

Train (Walker2d)

No DR:
```bash
python train.py --env_family walker2d --domain source --no-udr
```

UDR:
```bash
python train.py --env_family walker2d --domain source --udr --udr_ratio 0.20
```

With custom name:
```bash
python train.py --env_family walker2d --domain source --udr --udr_ratio 0.20 --run_name walker_udr_seed0
```

ADR (placeholder)

ADR is exposed as a flag, but must be implemented in the environment.
If --adr is enabled and ADR is not implemented, the environment should raise NotImplementedError
to avoid accidental “fake ADR” runs.

Example (will fail until ADR is implemented):
```bash
python train.py --env_family hopper --domain source --adr
```

Test a trained model (headless)

Rendering can be problematic on some Wayland/OpenGL setups.
The most reliable check is headless testing (no window):

Hopper, UDR run using the auto-generated default run id:

```bash
python test.py --env_family hopper --model models/hopper/source/udr_r0.20/ppo_lr0.0003_ns2048_nenv8_seed0_ts1000000.zip
```

If you used --run_name hopper_udr_seed0, the path is simpler:
```bash
python test.py --env_family hopper --model models/hopper/source/udr_r0.20/ppo_hopper_udr_seed0.zip
```

Walker2d example:
```bash
python test.py --env_family walker2d --model models/walker2d/source/udr_r0.20/ppo_lr0.0003_ns2048_nenv8_seed0_ts1000000.zip
```

By default, test.py runs both domains: source and target.
To test only one domain:
```bash
python test.py --env_family hopper --domains target --model <path-to-model.zip>
```

Test with rendering (optional)

Enable a human window with --render:
```bash
python test.py --env_family hopper --model <path-to-model.zip> --render
```

If rendering fails (OpenGL context errors), run without --render and use video recording
(see test_and_save_video.py) once your offscreen rendering backend is configured.

Useful notes

--domain in train.py indicates where the policy is trained (source or target).

train.py evaluates the trained policy on both source and target at the end of training.
The evaluation is run with DR disabled to keep comparisons fair.
