# tictactoe
Original `fit.py`, `env.py` and `play.py` was modified to support self-play.  
Agent uses dqn implementation from [pytorch examples](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### Installation 

```bash
conda create -n tictac python=3.10
conda activate tictac
cd tictactoe
pip install -r requirements.txt
```


### Evaluate

```
python -m tictactoe --self_play --file_name adversarial_1 --opponent_file_name adversarial_2  # to run agent vs agent
ptyhon -m tictactoe  # to play with human
```