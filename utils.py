from geografics import Distances
from typing import List

class Passenger:
    def __init__(self, id, origin, destination):
        self.id = id
        self.origin = origin
        self.destination = destination

class Group:
    def __init__(self, gid: int, passengers: List[Passenger]):
        self.id = gid
        self.passengers = passengers
        self.origin = passengers[0].origin
        self.destination = passengers[0].destination

    def __iter__(self):
        return iter(self.passengers)
    
class RechargePoint:
    def __init__(self, id, location):
        self.id = id
        self.location = location

class Vehicle:
    def __init__(self, id, start_location=(0, 0), battery=100.0, consumption_per_km=0.4, min_charge=20.0):
        self.id = id
        self.start_location = start_location
        self.battery = battery  # carga atual em %
        self.consumption_per_km = consumption_per_km  # porcentagem por km
        self.min_charge = min_charge  # nível mínimo antes de recarregar

    def distance_to(self, point, dist_func=Distances.euclidean):
        return dist_func(self.start_location[0], self.start_location[1], point[0], point[1])

    def energy_needed(self, distance):
        return distance * self.consumption_per_km

    def needs_recharge(self, distance):
        return self.battery - self.energy_needed(distance) < self.min_charge

if __name__ == "__main__":
    print("--- Verificando funcionalidades do Vehicle ---")
    origem = (0.0, 0.0)
    destino = (1.4, 10.4)

    carro = Vehicle(id=1, start_location=origem, battery=25.0, consumption_per_km=2.0, min_charge=20.0)
    distancia = carro.distance_to(destino)
    energia = carro.energy_needed(distancia)
    precisa_recarregar = carro.needs_recharge(distancia)

    print(f"Distância até o destino: {distancia:.2f} km")
    print(f"Energia necessária: {energia:.2f}%")
    print(f"Precisa recarregar? {'Sim' if precisa_recarregar else 'Não'}")
