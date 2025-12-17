from test_rl_agent import evaluate_agent
num_eps = [10000, 100000, 500000]
decay_rates = [0.9999, 0.99999, 0.999999]
train_opps = ["random_opp", "smarter_opp"]
for opp in train_opps:
    print("----------------"+opp+" training----------------")
    if opp == "random_opp":
        print("Trained against an opponent that plays moves randomly.")
    else:
        print("Trained against a tactical opponent that creates/blocks 3s and 2s and prefers center play.")
    for i in range(len(num_eps)):
        print(f"# of Training Episodes: {num_eps[i]}")
        print(f"Decay Rate: {decay_rates[i]}")
        filename = 'Q_table_'+str(num_eps[i])+'_'+str(decay_rates[i])+'_'+opp+'.pickle'
        evaluate_agent(filename, episodes=10000)