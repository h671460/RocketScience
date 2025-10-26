# WaterRocketDraft1.py
# Physically plausible water-rocket simulator with plots
# - Adiabatic air expansion while water is expelled
# - Orifice mass flow for water through the nozzle
# - Thrust = mdot*v_exit + pressure term
# - Ballistic coast with quadratic drag after water depletion
#
# Usage:
#   python WaterRocketDraft1.py
# or import WaterRocketSim from this file and use it on other scripts.

import math
import matplotlib.pyplot as plt


class WaterRocketSim:
    def __init__(self,
                 V_bottle_liters=0.52,     # total internal volume (L)
                 V_water0_liters=0.26,     # initial water volume (L)
                 P0_abs_bar=5.0,           # initial absolute pressure (bar). ~4 bar gauge + 1 bar atm
                 nozzle_diameter_mm=8.0,  # nozzle diameter (mm)
                 Cd_nozzle=0.9,            # nozzle discharge coefficient
                 m_dry=0.20,               # dry mass (kg): bottle + payload
                 T0_K=293.15,              # initial air temperature (K)
                 gamma=1.4,                # ratio of specific heats for air
                 Cd_drag=0.5,              # body drag coefficient
                 A_ref=0.005,              # frontal area (m^2) ~ 8 cm diameter
                 rho_air=1.225,            # air density (kg/m^3)
                 rho_water=1000.0,         # water density (kg/m^3)
                 g=9.81,                   # gravity (m/s^2)
                 dt=0.0005,                # time step (s)
                 t_max=10.0):              # safety cap (s)

        # Unit conversions and derived initial values
        self.V_bottle = V_bottle_liters * 1e-3      # m^3
        self.V_water0 = V_water0_liters * 1e-3      # m^3
        self.V_air0 = self.V_bottle - self.V_water0 # m^3
        self.P0 = P0_abs_bar * 1e5                  # Pa (absolute   (for now))
        self.p_atm = 1.013e5                        # Pa
        self.noz_A = math.pi * (nozzle_diameter_mm * 1e-3 / 2.0) ** 2  # m^2

        # Store parameters
        self.Cd_nozzle = Cd_nozzle
        self.m_dry = m_dry
        self.T0 = T0_K
        self.gamma = gamma
        self.Cd_drag = Cd_drag
        self.A_ref = A_ref
        self.rho_air = rho_air
        self.rho_water = rho_water
        self.g = g
        self.dt = dt
        self.t_max = t_max

        # Initial air mass from ideal gas law
        R_air = 287.05  # J/(kg*K)
        self.m_air0 = self.P0 * self.V_air0 / (R_air * self.T0)

        self.reset()

    def reset(self):
        # State
        self.t = 0.0
        self.V_air = self.V_air0
        self.V_water = self.V_bottle - self.V_air
        self.m_water = self.V_water * self.rho_water
        self.m_air = self.m_air0     # assume constant during water phase
        self.m = self.m_dry + self.m_water + self.m_air

        # Kinematics
        self.h = 0.0
        self.v = 0.0

        # Adiabatic constant: p * V^gamma = const
        self.Kadi = self.P0 * (self.V_air0 ** self.gamma)

        # History buffers
        self.T = []
        self.H = []
        self.VV = []
        self.P = []
        self.TH = []
        self.M = []
        self.MW = []

        self.thrusting = True
        self.apogee_index = None

    def step(self):
        """Integrate one time step."""
        # Water expulsion phase
        if self.thrusting and self.V_air < self.V_bottle and self.m_water > 0:
            p_internal = self.Kadi / (self.V_air ** self.gamma)
            delta_p = max(p_internal - self.p_atm, 0.0)

            if delta_p <= 0.0 or self.noz_A <= 0.0:
                thrust = 0.0
                mdot_w = 0.0
                self.thrusting = False
            else:
                # Exit speed (Bernoulli/orifice)
                v_exit = math.sqrt(2.0 * delta_p / self.rho_water)
                # Mass flow through nozzle
                mdot_w = self.Cd_nozzle * self.rho_water * self.noz_A * v_exit
                # Pressure thrust (small near ambient, but include)
                F_pressure = (p_internal - self.p_atm) * self.noz_A
                thrust = mdot_w * v_exit + F_pressure

                # Update volumes/masses
                dV_water = (mdot_w / self.rho_water) * self.dt
                self.V_air = min(self.V_bottle, self.V_air + dV_water)
                self.V_water = max(0.0, self.V_bottle - self.V_air)

                dm_water = mdot_w * self.dt
                self.m_water = max(0.0, self.m_water - dm_water)
                self.m = self.m_dry + self.m_water + self.m_air

                if self.m_water <= 0.0 or self.V_air >= self.V_bottle:
                    self.thrusting = False
        else:
            # Coast (no thrust)
            p_internal = self.p_atm
            thrust = 0.0
            mdot_w = 0.0
            self.thrusting = False

        # Aerodynamic drag
        drag_mag = 0.5 * self.rho_air * self.Cd_drag * self.A_ref * (self.v ** 2)
        drag = -drag_mag if self.v > 0 else drag_mag

        # Net acceleration
        a = (thrust + drag - self.m * self.g) / self.m

        # Integrate kinematics
        self.v += a * self.dt
        self.h = max(0.0, self.h + self.v * self.dt)

        # Store history
        self.T.append(self.t)
        self.H.append(self.h)
        self.VV.append(self.v)
        self.P.append(p_internal)
        self.TH.append(thrust)
        self.M.append(self.m)
        self.MW.append(self.m_water)

        self.t += self.dt

    def run(self):
        self.reset()
        apogee_found = False
        while self.t < self.t_max:
            self.step()
            # Apogee detection: first time after thrust end where v <= 0
            if (not self.thrusting) and (self.v <= 0) and (not apogee_found):
                self.apogee_index = len(self.T) - 1
                apogee_found = True
                # continue a bit more for smoother plot
                if self.t > 2.0:
                    break
        return {
            "t": self.T, "h": self.H, "v": self.VV, "p": self.P,
            "thrust": self.TH, "m": self.M, "m_water": self.MW,
            "apogee_index": self.apogee_index
        }

    def plot_thrust(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.T, self.TH)
        plt.xlabel("Time (s)")
        plt.ylabel("Thrust (N)")
        plt.title("Water Rocket Thrust vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_height(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.T, self.H)
        if self.apogee_index is not None:
            ta = self.T[self.apogee_index]
            ha = self.H[self.apogee_index]
            plt.scatter([ta], [ha])
            plt.annotate(f"Apogee ~ {ha:.1f} m at {ta:.2f} s",
                         (ta, ha), xytext=(10, 10), textcoords="offset points")
        plt.xlabel("Time (s)")
        plt.ylabel("Height (m)")
        plt.title("Water Rocket Height vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = WaterRocketSim(
        V_bottle_liters=0.52,
        V_water0_liters=0.26,
        P0_abs_bar=5.0,           # 4 bar gauge + 1 bar atm
        nozzle_diameter_mm=20.0,
        Cd_nozzle=0.9,
        m_dry=0.20,
        T0_K=293.15,
        gamma=1.4,
        Cd_drag=0.5,
        A_ref=0.005,
        rho_air=1.225,
        rho_water=1000.0,
        g=9.81,
        dt=0.0005,
        t_max=10.0
    )
    res = sim.run()

    # Quick metrics
    thrust = res["thrust"]
    peak_thrust = max(thrust) if thrust else 0.0
    t = res["t"]
    t_burnout = next((t[i] for i in range(1, len(thrust)) if thrust[i] == 0.0 and thrust[i-1] > 0.0), None)
    ap_i = res["apogee_index"]
    ap_h = res["h"][ap_i] if ap_i is not None else max(res["h"])
    ap_t = t[ap_i] if ap_i is not None else t[res["h"].index(ap_h)]

    print(f"Peak thrust: {peak_thrust:.1f} N")
    print(f"Burnout time: {t_burnout:.3f} s" if t_burnout is not None else "Burnout time not detected")
    print(f"Apogee: {ap_h:.2f} m at t = {ap_t:.2f} s")

    # Plots
    sim.plot_thrust()
    sim.plot_height()