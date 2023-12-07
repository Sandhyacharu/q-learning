### EX NO: 07
### DATE:
# <p align="center">Q Learning Algorithm</p>

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.

### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3:
After training, derive the optimal policy from the Q-table.

### Step 4:
Implement the Monte Carlo method to estimate state values.

### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
```python3
# Developed by: Sandhya Charu N
# Register Number: 212220230041

def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```


## OUTPUT:
### Mention the optimal policy, optimal value function , success rate for the optimal policy.
![image](https://github.com/Sandhyacharu/q-learning/assets/75235167/5054350d-40a3-45ca-946f-a091b2dac17b)

![image](https://github.com/Sandhyacharu/q-learning/assets/75235167/dd70f5b3-835d-452e-8375-10215e6ac3cb)

### Include plot comparing the state value functions of Monte Carlo method and Qlearning.
![image](https://github.com/Sandhyacharu/q-learning/assets/75235167/a865ac59-ed02-447b-8bf3-cae65093212f)

![image](https://github.com/Sandhyacharu/q-learning/assets/75235167/37fbfe17-cf6d-4583-9714-a37aeaf3fef8)

## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
