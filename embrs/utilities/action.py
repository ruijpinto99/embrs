from embrs.utilities.fire_util import CellStates, FireTypes

class Action:
    def __init__(self, time, x, y):
        self.time = time
        self.loc = (x, y)

class SetFuelMoisture(Action):
    def __init__(self, time, x, y, moisture):
        super().__init__(time, x, y)
        self.moisture = moisture

    def perform(self, fire):
        fire.set_fuel_moisture_at_xy(self.loc[0], self.loc[1], self.moisture)
        
class SetFuelContent(Action):
    def __init__(self, time, x, y, fuel_content):
        super().__init__(time, x, y)
        self.content = fuel_content

    def perform(self, fire):
        fire.set_fuel_content_at_xy(self.loc[0], self.loc[1], self.content)

class SetIgnition(Action):
    def __init__(self, time, x, y, fire_type):
        super().__init__(time, x, y)
        self.fire_type = fire_type

    def perform(self, fire):
        fire.set_state_at_xy(self.loc[0], self.loc[1], CellStates.FIRE, fire_type=FireTypes.WILD)

