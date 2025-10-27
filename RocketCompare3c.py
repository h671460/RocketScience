import math
import matplotlib.pyplot as plt


class WaterRocketSim:
    """
    Simple 1D vertical water-rocket model:
    - Adiabatic air expansion while water is expelled
    - Liquid-only thrust phase, ends when water is gone or pressure ~ atm
    - After burnout: pure ballistic coast w/ quadratic drag
    - No 'air jet' phase after water is gone (deliberately omitted here)
    """

    def __init__(self,
                 V_bottle_liters=0.52,
                 V_water0_liters=0.26,
                 P0_abs_bar=5.0,         # absolute, e.g. 5 bar = ~4 bar gauge
                 nozzle_diameter_mm=10.0,
                 Cd_nozzle=0.9,          # discharge coefficient
                 m_dry=0.20,             # kg, empty rocket + payload
                 T0_K=293.15,
                 gamma=1.4,
                 Cd_drag=0.5,
                 A_ref=0.005,            # m^2
                 rho_air=1.225,
                 rho_water=1000.0,
                 g=9.81,
                 dt=0.0005,
                 t_max=10.0):

        # Geometry / initial volumes
        self.V_bottle = V_bottle_liters * 1e-3         # m^3
        self.V_water0 = V_water0_liters * 1e-3         # m^3
        self.V_air0 = self.V_bottle - self.V_water0    # m^3

        # Thermo
        self.P0 = P0_abs_bar * 1e5                     # Pa
        self.p_atm = 1.013e5                           # Pa
        self.T0 = T0_K
        self.gamma = gamma
        self.R_air = 287.05                            # J/(kg*K)

        # Nozzle
        self.noz_A = math.pi * (nozzle_diameter_mm * 1e-3 / 2.0) ** 2
        self.Cd_nozzle = Cd_nozzle

        # Mass and dynamics parameters - LEGG til denne
        self.m_dry = m_dry
        
        # Aero / dynamics
        self.Cd_drag = Cd_drag
        self.A_ref = A_ref
        self.rho_air = rho_air
        self.rho_water = rho_water
        self.g = g
        self.dt = dt
        self.t_max = t_max

        # Initial masses
        # ideal gas: pV = m R T -> m_air0
        self.m_air0 = self.P0 * self.V_air0 / (self.R_air * self.T0)
        self.m_air = self.m_air0
        self.m_water0 = self.V_water0 * self.rho_water
        self.m_water = self.m_water0

        # Will be updated during reset()
        self.reset()

    def reset(self):
        # Time
        self.t = 0.0

        # Volumes / masses
        self.V_air = self.V_air0
        self.V_water = self.V_bottle - self.V_air
        self.m_water = self.V_water * self.rho_water
        self.m_air = self.m_air0  # stays constant during water phase in this simple model
        self.m = self.m_dry + self.m_water + self.m_air

        # Kinematics
        self.h = 0.0
        self.v = 0.0

        # Adiabatic invariant p V^gamma = const
        self.Kadi = self.P0 * (self.V_air0 ** self.gamma)

        # Histories
        self.T_hist = []
        self.H_hist = []
        self.V_hist = []
        self.P_hist = []
        self.F_hist = []
        self.M_hist = []
        self.Mw_hist = []

        # State flags
        self.thrusting = True
        self.burnout_index = None
        self.apogee_index = None

    def _liquid_thrust_step(self):
        """
        Compute thrust while there is still water and pressure > atm.
        Update internal state (volumes, masses, total mass).
        Returns (thrust, p_internal). If no thrust possible, returns (0, p_atm).
        """
        # Internal pressure from adiabatic law
        p_internal = self.Kadi / (self.V_air ** self.gamma)
        delta_p = max(p_internal - self.p_atm, 0.0)

        if (not self.thrusting) or (self.m_water <= 0.0) or (delta_p <= 0.0):
            self.thrusting = False
            return 0.0, p_internal

        # Exit speed of the water jet (Bernoulli)
        # v_exit ~ sqrt(2 Δp / rho)
        v_exit_core = math.sqrt(2.0 * delta_p / self.rho_water)

        # We apply Cd_nozzle to the mass flow, not to the speed twice.
        # Mass flow rate through nozzle:
        mdot_w = self.Cd_nozzle * self.rho_water * self.noz_A * v_exit_core

        # Momentum thrust
        F_momentum = mdot_w * v_exit_core

        # Pressure thrust term
        F_pressure = delta_p * self.noz_A

        thrust = F_momentum + F_pressure

        # Update liquid volume and rocket mass
        dV_water = (mdot_w / self.rho_water) * self.dt
        self.V_air = min(self.V_bottle, self.V_air + dV_water)
        self.V_water = max(0.0, self.V_bottle - self.V_air)

        dm_water = mdot_w * self.dt
        if dm_water > self.m_water:
            dm_water = self.m_water
        self.m_water = max(0.0, self.m_water - dm_water)

        # Update total mass (air fixed in this simplified model)
        self.m = self.m_dry + self.m_water + self.m_air

        # Burnout detection: if we just ran out of water or volume fully air
        if self.m_water <= 0.0 or self.V_air >= self.V_bottle:
            self.thrusting = False

        return thrust, p_internal

    def step(self):
        """
        Advance simulation by one dt:
        - Compute thrust (if any)
        - Apply drag + gravity
        - Euler integrate velocity and height
        - Log state
        """
        thrust, p_internal = self._liquid_thrust_step()

        # Drag force magnitude
        drag_mag = 0.5 * self.rho_air * self.Cd_drag * self.A_ref * (self.v ** 2)
        drag = -drag_mag if self.v > 0 else drag_mag  # opposite velocity direction

        # Net accel
        a = (thrust + drag - self.m * self.g) / self.m

        # Integrate kinematics
        self.v += a * self.dt
        self.h = max(0.0, self.h + self.v * self.dt)

        # Save history
        self.T_hist.append(self.t)
        self.H_hist.append(self.h)
        self.V_hist.append(self.v)
        self.P_hist.append(p_internal)
        self.F_hist.append(thrust)
        self.M_hist.append(self.m)
        self.Mw_hist.append(self.m_water)

        # Detect burnout index = first instant thrust goes to (or stays at) 0
        if self.burnout_index is None:
            if len(self.F_hist) > 1 and self.F_hist[-2] > 0.0 and self.F_hist[-1] == 0.0:
                self.burnout_index = len(self.F_hist) - 1

        # Detect apogee: only after burnout, when vertical v crosses downwards
        if (self.apogee_index is None) and (self.burnout_index is not None):
            if self.v <= 0.0:
                self.apogee_index = len(self.T_hist) - 1

        # Time marches
        self.t += self.dt

    def run(self):
        self.reset()
        while self.t < self.t_max:
            self.step()

            # Stop early if:
            # - we've already passed apogee and coasted ~2s
            if (self.apogee_index is not None and
               self.t > self.T_hist[self.apogee_index] + 2.0):
                break

            # - rocket is back on ground after launch
            if self.h <= 0.0 and self.t > 0.05 and self.v < 0:
                break

        # Fallback burnout index if not caught by slope
        if self.burnout_index is None:
            active = [i for i, F in enumerate(self.F_hist) if F > 0]
            if active:
                self.burnout_index = active[-1]

        return {
            "t": self.T_hist,
            "h": self.H_hist,
            "v": self.V_hist,
            "thrust": self.F_hist,
            "pressure": self.P_hist,
            "mass": self.M_hist,
            "m_water": self.Mw_hist,
            "burnout_index": self.burnout_index,
            "apogee_index": self.apogee_index
        }

    # ---------------- ANALYTIC APPROX ----------------

    def analytic_tb(self):
        """
        Approximate burn time:
        Assume roughly constant average pressure in the tank,
        giving constant-ish mdot.
        """
        # avg internal absolute pressure ~ midpoint between P0 and atm
        P_avg = 0.5 * (self.P0 + self.p_atm)
        delta_p_avg = max(P_avg - self.p_atm, 0.0)

        if delta_p_avg <= 0:
            return 0.0

        # exit speed for water jet (no Cd on v here; Cd applies to mdot)
        v_exit_avg = math.sqrt(2.0 * delta_p_avg / self.rho_water)

        # mass flow ≈ Cd * rho * A * v_exit
        mdot_avg = self.Cd_nozzle * self.rho_water * self.noz_A * v_exit_avg

        m_water_total = self.m_water0  # all initial propellant
        if mdot_avg > 0:
            tb = m_water_total / mdot_avg
        else:
            tb = 0.0
        return tb

    def analytic_impulse(self):
        """
        Approximate total impulse = avg thrust * burn time.
        avg thrust = momentum term + pressure term using avg Δp.
        """
        tb = self.analytic_tb()
        if tb == 0:
            return 0.0

        P_avg = 0.5 * (self.P0 + self.p_atm)
        delta_p_avg = max(P_avg - self.p_atm, 0.0)

        v_exit_avg = math.sqrt(2.0 * delta_p_avg / self.rho_water)
        mdot_avg = self.Cd_nozzle * self.rho_water * self.noz_A * v_exit_avg

        F_mom_avg = mdot_avg * v_exit_avg
        F_press_avg = delta_p_avg * self.noz_A
        F_avg = F_mom_avg + F_press_avg

        I = F_avg * tb
        return I

    def analytic_burnout_velocity(self):
        """
        v_burnout ≈ (Impulse / m_avg) - g * t_b
        where m_avg is average rocket mass during burn
        (dry mass + air mass + ~half the water).
        """
        tb = self.analytic_tb()
        I = self.analytic_impulse()
        if tb == 0.0:
            return 0.0

        m_avg = self.m_dry + self.m_air0 + 0.5 * self.m_water0
        if m_avg <= 0:
            return 0.0

        vb = I / m_avg - self.g * tb
        return vb

    def analytic_height_nodrag(self):
        """
        Predict max height ignoring drag.
        Split into:
        - height gained during burn (approx constant accel from 0->vb)
        - coast height after burn (vb^2 / (2g))
        For h_burn we use s = 0.5 * a * t^2 with a ~ (I/t_b)/m_avg - g
        """
        tb = self.analytic_tb()
        vb = self.analytic_burnout_velocity()
        if tb == 0.0:
            return 0.0

        # approximate average thrust = I/tb
        I = self.analytic_impulse()
        F_avg = I / tb if tb > 0 else 0.0
        m_avg = self.m_dry + self.m_air0 + 0.5 * self.m_water0

        # avg accel during burn:
        if m_avg > 0:
            a_avg = (F_avg / m_avg) - self.g
        else:
            a_avg = 0.0

        # distance during burn ~ 0.5 * a_avg * tb^2
        h_burn = 0.5 * a_avg * (tb ** 2)

        # coast height:
        h_coast = (vb ** 2) / (2.0 * self.g) if vb > 0 else 0.0

        return h_burn + h_coast


def compare_and_print(sim: WaterRocketSim, do_plots=True):
    res = sim.run()

    t_arr = res["t"]
    h_arr = res["h"]
    v_arr = res["v"]
    F_arr = res["thrust"]
    i_burn = res["burnout_index"]
    i_ap = res["apogee_index"]

    # ----- numeric measurements -----
    dt = sim.dt
    if i_burn is not None and i_burn < len(t_arr):
        tb_num = t_arr[i_burn]
        vb_num = v_arr[i_burn]
        I_num = sum(F_arr[i] * dt for i in range(i_burn + 1))
    else:
        # fallback: whole array
        tb_num = t_arr[-1]
        vb_num = v_arr[-1]
        I_num = sum(F * dt for F in F_arr)

    hmax_num = max(h_arr) if h_arr else 0.0

    # apogee numeric
    if i_ap is not None and i_ap < len(h_arr):
        h_ap_num = h_arr[i_ap]
        t_ap_num = t_arr[i_ap]
    else:
        h_ap_num = hmax_num
        t_ap_num = t_arr[h_arr.index(hmax_num)] if h_arr else 0.0

    # ----- analytic estimates -----
    tb_an = sim.analytic_tb()
    I_an = sim.analytic_impulse()
    vb_an = sim.analytic_burnout_velocity()
    h_an = sim.analytic_height_nodrag()

    # utility for table formatting
    def fmt(x):
        return f"{x:.4g}" if x is not None else "n/a"

    def pctdiff(a, b):
        if a is None or a == 0:  # avoid div by zero
            return "n/a"
        return f"{abs(a - b)/abs(a)*100:.1f}%"

    print("=== Water Rocket Simulation Results ===\n")
    print("Analytic vs Numeric Comparison:")
    print(f"{'Parameter':<22} {'Analytic':<15} {'Numeric':<15} {'Diff %':<10}")
    print("-" * 65)

    print(f"{'Burnout time (s)':<22} {fmt(tb_an):<15} {fmt(tb_num):<15} {pctdiff(tb_an, tb_num):<10}")
    print(f"{'Impulse (N·s)':<22} {fmt(I_an):<15} {fmt(I_num):<15} {pctdiff(I_an, I_num):<10}")
    print(f"{'Burnout vel (m/s)':<22} {fmt(vb_an):<15} {fmt(vb_num):<15} {pctdiff(vb_an, vb_num):<10}")
    print(f"{'Max height (m)':<22} {fmt(h_an):<15} {fmt(h_ap_num):<15} {'(drag)':<10}")
    print("\nNote: Analytic max height ignores drag. Numeric height includes drag, so numeric will be lower.\n")

    if do_plots:
        # 1) Thrust vs Time
        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 2, 1)
        # Thrust vs Time - Numeric only (analytic uses average)
        ax1.plot(t_arr, F_arr, 'b-', linewidth=2, label='Numeric')
        if i_burn is not None and i_burn < len(t_arr):
            ax1.axvline(t_arr[i_burn], color='r', linestyle='--', label=f'Burnout @ {t_arr[i_burn]:.3f}s (num)')
        if tb_an > 0:
            ax1.axvline(tb_an, color='orange', linestyle='--', label=f'Burnout @ {tb_an:.3f}s (analytic)')
        
        # Zoom in on thrust phase only
        t_limit = max(tb_num if tb_num else tb_an, tb_an) * 1.5
        ax1.set_xlim(0, t_limit)
        
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Thrust (N)")
        ax1.set_title("Thrust vs Time")
        ax1.grid(True, alpha=0.3)

        # 2) Height vs Time
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(t_arr, h_arr, 'g-', linewidth=2, label='Height numeric (drag)')
        ax2.axhline(h_an, color='orange', linestyle='--', linewidth=2,
                    label=f'Analytic no-drag h_max={h_an:.2f} m')
        if i_ap is not None and i_ap < len(t_arr):
            ax2.scatter([t_arr[i_ap]], [h_arr[i_ap]], color='r', s=80, zorder=5)
            ax2.annotate(f"Apogee num {h_arr[i_ap]:.2f} m",
                         xy=(t_arr[i_ap], h_arr[i_ap]),
                         xytext=(10, -20),
                         textcoords="offset points",
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         arrowprops=dict(arrowstyle='->'))
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Height (m)")
        ax2.set_title("Height vs Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3) Velocity vs Time
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(t_arr, v_arr, 'purple', linewidth=2, label='Velocity numeric')
        if i_burn is not None and i_burn < len(t_arr):
            ax3.axvline(t_arr[i_burn], color='r', linestyle='--', alpha=0.5)
            ax3.scatter([t_arr[i_burn]], [v_arr[i_burn]], color='r', s=60,
                        label=f'vb_num={v_arr[i_burn]:.2f} m/s')
        if vb_an > 0:
            ax3.axhline(vb_an, color='orange', linestyle='--', linewidth=1,
                        label=f'vb_an={vb_an:.2f} m/s')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity (m/s)")
        ax3.set_title("Velocity vs Time")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4) Impulse bar chart
        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(['Analytic', 'Numeric'], [I_an, I_num],
                alpha=0.7)
        ax4.set_ylabel("Impulse (N·s)")
        diff_txt = pctdiff(I_an, I_num)
        ax4.set_title(f"Total Impulse\nDiff: {diff_txt}")
        ax4.grid(True, alpha=0.3, axis='y')
        for i_bar, val in enumerate([I_an, I_num]):
            ax4.text(i_bar, val, f'{val:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = WaterRocketSim(
        V_bottle_liters=0.52,
        V_water0_liters=0.26,
        P0_abs_bar=5.0,         # absolute pressure in bar
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

    compare_and_print(sim, do_plots=True)