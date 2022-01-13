class Countries():
    def __init__(self, capital, population):
        self.capital = capital
        self.population = population

if __name__ == "__main__":
    obj = Countries("delhi", [10,20,30])
    print(obj.population)
