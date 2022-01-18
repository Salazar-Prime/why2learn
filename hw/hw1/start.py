"""
Homework 1: Python practice
Author: Varun Aggarwal
Last Modified: 17 Jan 2022
"""


class Countries:
    def __init__(self, capital, population):
        self.capital = capital
        self.population = population

    def net_population(self):
        # birth - death + last_count
        current_net = self.population[0] - self.population[1] + self.population[2]
        return current_net


class GeoCountry(Countries):
    def __init__(self, capital, population, area, density=0):
        Countries.__init__(self, capital, population)
        self.area = area
        self.density = density

    def density_calculator1(self):
        if len(self.population) == 3:
            # calling parent function
            self.density = Countries.net_population(self) / self.area
        else:
            self.density = self.net_population() / self.area

    def density_calculator2(self):
        """
        update population, call density_calculator1
        new_last_count =  last_count +death - birth
        """
        if len(self.population) == 3:
            self.population[2] = (
                self.population[2] + self.population[1] - self.population[0]
            )
        else:
            self.population[3] = (
                self.population[3] + self.population[1] - self.population[0]
            )
        self.density_calculator1()

    def net_density(self, choice):
        if choice == 1:
            return self.density_calculator1
        elif choice == 2:
            return self.density_calculator2
        else:
            print("Incorrect Choice")
            return

    # TASK 7: add net_population
    # let it handle population length = 4
    def net_population(self):
        # only do this when length is 3
        if len(self.population) == 3:
            self.population.append(Countries.net_population(self))
        current_net = (
            self.population[0]
            - self.population[1]
            + (self.population[2] + self.population[3]) / 2
        )
        return current_net


if __name__ == "__main__":
    # initialized, never used
    obj1 = Countries("Piplipol", [40, 30, 20])
    obj2 = GeoCountry("Polpip", [55, 10, 70], 230)

    # test case
    ob1 = GeoCountry("YYY", [20, 100, 1000], 5)
    print(ob1.density)  # 0
    print(ob1.population)  # [20,100,1000]
    fn = ob1.net_density(1)
    fn()
    print(ob1.density)  # 184.0
    fn = ob1.net_density(2)
    fn()
    print(ob1.population)  # [20, 100, 1080]
    print(ob1.density)  # 200.0
    ob2 = GeoCountry("ZZZ", [20, 50, 100], 12)
    fun = ob2.net_density(2)
    print(ob2.density)  # 0
    fun()
    print("{:.2f}".format(ob2.density))  # 8.33
    print(ob1.population)  # [20,100, 1080]
    print(ob1.net_population())  # 960.0
    print(ob1.population)  # [20,100,1080,1000]
    print(ob1.density)  # 200.0
    ob1.density_calculator1()
    print(ob1.population)  # [20, 100, 1080, 1000]
    print(ob1.density)  # 192.0
    ob1.density_calculator2()
    print(ob1.population)  # [20, 100, 1080, 1080]
    print(ob1.density)  # 200
