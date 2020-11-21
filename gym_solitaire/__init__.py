from gym.envs.registration import register

register(
    id='gym_solitaire-v0',
    entry_point='gym_solitaire.envs:SolitaireEnv'
)
