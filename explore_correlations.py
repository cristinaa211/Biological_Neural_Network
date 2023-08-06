import pandas as pd
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    df = pd.read_csv("./correlation_list.csv")
    df['neuron_pair'] = df['inh_type'] + '-' + df['exc_type']
    print(df.info())
    df.pivot_table(index = 'neuron_pair', columns= 'no_neurons', values = 'correlation_coeff').plot.bar(stacked = True)
    plt.show()
    fig = plt.figure()
    plt.plot(df['correlation_coeff'])
    plt.show()

