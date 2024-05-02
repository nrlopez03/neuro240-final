# Impact of Changing Transition Function in Deep RL

## Motivation
This paper explores the Out-of-Distribution (OOD) generalization problem as it pertains to Deep Q-Learning agents. In the OOD generalization problem, an agent is tested using data or an environment that is different from that upon which it was trained. Various effects, such as the quantity of training time/data and exact environment/data upon which the agent was trained, determine how well the agent performs in the unique tests. 

## Approach
To examine this problem, we trained a Deep Q-Learning agent to play the game Pac-Man, modified to be played on a slightly smaller layout. To create differences in the training and testing environments, we altered the ghost transition functions. In particular, four classes of ghost agents were created, each of varying levels of aggression and awareness. Furthermore, various layer counts were used for the DQN, as we aimed to examine potential differences in performance across this axis. Thus, while we focus primarily on the former, our research question is two fold. We examine how changing ghost movement probabilities to be different in testing than it was in training affects the performance of the reinforcement learning agent. We further examine how the number of layers used affects agent performance within across the changes in transition function. 

## Running the Code
To run the code, follow these steps:
1. Select a load file or save file in *pacmanDQN_Agents.py*. 
2. Specify the number of layers to be used by changing the DQN file imported into *pacmanDQN_Agents.py*.
3. If logging the results, change the following parameters in *pacman.py* to activate logging and track the desired variables: set {{logging}} to true, {{layers}} to the number of layers used in the DQN, and {{trainGames}} to the total number of games to be used during training. 
4. Run the following command:

 ```bash
 python3.8 pacman.py -p PacmanDQN -n numGames -x trainGames -l smallGrid -g trainGhost -s testGhost 
 ```
where {{numGames}} is the total number of games to be played, {{trainGames}} is the number of training games, {{trainGhost}} is the class of ghost to be used during training, and {{testGhost}} is the class of ghost to be used during testing. 

## Acknowledgements

Application of DQN Framework to Pac-Man by Tycho van der Ouderaa:
* [van der Ouderaa, Tycho (2016). Deep Reinforcement Learning in Pac-man.](https://moodle.umons.ac.be/pluginfile.php/404484/mod_folder/content/0/Pacman_DQN.pdf)

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))