from geografics import Distances
from typing import List, Tuple

class Passenger:
    def __init__(self, id, origin, destination):
        self._id = id
        self._origin = origin
        self._destination = destination
    
    @property
    def id(self):
        return self._id
    
    def distance(self, metric: str="euclidean") -> float:
        metric_fn = getattr(Distances, metric)
        return metric_fn(*self._origin, *self._destination)
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def destination(self):
        return self._destination
    
    @property
    def str_origin(self):
        return f"Po{self.id}"
    
    @property
    def str_destination(self):
        return f"Pd{self.id}"
    

class Group:
    def __init__(self, gid: int, passengers: List[Passenger]):
        self._id = gid
        self.passengers = passengers
        self._origin = passengers[0].origin
        self._destination = passengers[0].destination

    def __iter__(self):
        return iter(self.passengers)    
    
    @property
    def id(self):
        return self._id
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def destination(self):
        return self._destination
    
    @property
    def str_origin(self):
        return f"Go{self.id}"
    
    @property
    def str_destination(self):
        return f"Gd{self.id}"
    
    
class RechargePoint:
    def __init__(self, id, location):
        self._id = id
        self._location = location
    
    @property
    def id(self):
        return self._id
    
    @property
    def location(self):
        return self._location
    
    @property
    def str_location(self):
        return f"R{self.id}"
    
class Depot:
    def __init__(self, 
                 id: int, 
                 location: Tuple[float, float]=(0.0, 0.0)
                ):
        self._id = id 
        self._location = location

    @property
    def id(self):
        return self._id
    
    @property
    def location(self):
        return self._location
    
    @property
    def str_location(self):
        return f"D{self.id}"

    

class Vehicle:
    def __init__(self, id, start_location=(0, 0), battery=100.0, consumption_per_km=0.3, min_charge=20.0):
        self._id = id
        self._start_location = start_location
        self._battery = battery  # carga atual em %
        self._consumption_per_km = consumption_per_km  # porcentagem por km
        self._min_charge = min_charge  # nível mínimo antes de recarregar

    @property
    def id(self):
        return self._id 
    
    @property
    def start_location(self):
        return self._start_location
    
    @property
    def battery(self):
        return self._battery
    
    @property
    def consumption_per_km(self):
        return self._consumption_per_km
    
    @property
    def min_charge(self):
        return self._min_charge

    def distance_to(self, point: Tuple[float, float]=(0.0,0.0), dist_func=Distances.euclidean):
        return dist_func(*self._start_location, *point)

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
