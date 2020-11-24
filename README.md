# Local Installation

```
$ pip3 install -e .
Obtaining file:///Users/administrator/HomeProjects/gym-solitaire
Requirement already satisfied: gym in /usr/local/lib/python3.9/site-packages (from gym-solitaire==0.0.1) (0.17.3)
Requirement already satisfied: scipy in /usr/local/lib/python3.9/site-packages (from gym->gym-solitaire==0.0.1) (1.5.4)
Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.9/site-packages (from gym->gym-solitaire==0.0.1) (1.5.0)
Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.9/site-packages (from gym->gym-solitaire==0.0.1) (1.6.0)
Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.9/site-packages (from gym->gym-solitaire==0.0.1) (1.19.4)
Requirement already satisfied: future in /usr/local/lib/python3.9/site-packages (from pyglet<=1.5.0,>=1.4.0->gym->gym-solitaire==0.0.1) (0.18.2)
Installing collected packages: gym-solitaire
  Running setup.py develop for gym-solitaire
Successfully installed gym-solitaire
```

# TODO

* ~~implement `render` for mode 'human'~~
* implement `render` for mode 'rgb_array'
  * for an example, see:
    * https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L161 
    * https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
* add class/method comments
* ~~add unit tests~~

# Links

* [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
