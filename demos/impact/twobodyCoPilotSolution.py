"""
Two-Body Relaxation Time Numerical Experiment

This module implements a numerical simulation to study gravitational relaxation
of a test particle moving through a medium of field particles. The experiment
tests the time it takes for cumulative velocity changes to equal the initial velocity.

Author: AI Assistant (GitHub Copilot)
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time


def impulse_approximation(v_rel, b, M, G):
    """
    Calculate velocity change from gravitational encounter using impulse approximation.
    
    Parameters:
    -----------
    v_rel : float
        Relative velocity between test particle and field particle
    b : float
        Impact parameter
    M : float
        Mass of field particle
    G : float
        Gravitational constant
        
    Returns:
    --------
    delta_v : float
        Magnitude of velocity change
    """
    # Avoid division by zero
    if b == 0:
        b = 1e-10
    
    # Gravitational focusing parameter
    p = G * M / (b * v_rel**2)
    
    # Scattering angle (small angle approximation for p << 1)
    if p < 0.1:
        theta = 2 * p  # Small angle approximation
    else:
        theta = 2 * np.arctan(p)  # Full expression
    
    # Velocity change magnitude
    delta_v = v_rel * theta
    
    return delta_v


def generate_encounter(R,nparticles):
    """
    Generate random encounter geometry within cylinder.
    
    Parameters:
    -----------
    R : float
        Cylinder radius
        
    Returns:
    --------
    b : float
        Impact parameter (0 to R)
    phi : float
        Azimuthal angle (0 to 2π)
    """
    # Impact parameter - uniform distribution in area gives sqrt(random)
    b = R * np.sqrt(np.random.random(nparticles))
    
    # Azimuthal angle - uniform distribution
    phi = 2 * np.pi * np.random.random(nparticles)
    
    return b, phi


def apply_velocity_change(velocity_vector, delta_v, phi):
    """
    Apply velocity change to test particle in random perpendicular direction.
    
    Parameters:
    -----------
    velocity_vector : array
        Current velocity vector [vx, vy, vz]
    delta_v : float
        Magnitude of velocity change
    phi : float
        Azimuthal angle for random direction
        
    Returns:
    --------
    new_velocity : array
        Updated velocity vector
    """
    # Assume motion primarily in z-direction
    # Apply perturbation in x-y plane
    delta_vx = delta_v * np.cos(phi)
    delta_vy = delta_v * np.sin(phi)
    
    # Small longitudinal component (energy conservation approximately)
    delta_vz = -0.1 * delta_v * np.random.choice([-1, 1])
    
    new_velocity = velocity_vector + np.array([delta_vx, delta_vy, delta_vz])
    
    return new_velocity


def theoretical_relaxation_time(v0, n, M, m, G, R):
    """
    Calculate theoretical relaxation time using Chandrasekhar's formula.
    
    Parameters:
    -----------
    v0 : float
        Initial velocity
    n : float
        Number density of field particles
    M : float
        Mass of field particles
    m : float
        Mass of test particle
    G : float
        Gravitational constant
    R : float
        Maximum impact parameter (cylinder radius)
        
    Returns:
    --------
    t_relax : float
        Theoretical relaxation time
    """
    # Coulomb logarithm (gravitational)
    Lambda = R * v0**2 / (G * M)
    ln_Lambda = np.log(max(Lambda, 2.0))  # Ensure ln_Lambda > 0
    
    # Relaxation time (Chandrasekhar formula)
    t_relax = v0**3 / (4 * np.pi * G**2 * M**2 * n * ln_Lambda)
    
    return t_relax


def run_single_timestep(velocity_vector, dt, n, R, M, G):
    """
    Evolve system for one time step.
    
    Parameters:
    -----------
    velocity_vector : array
        Current velocity vector
    dt : float
        Time step
    n : float
        Number density
    R : float
        Cylinder radius
    M : float
        Field particle mass
    G : float
        Gravitational constant
        
    Returns:
    --------
    new_velocity : array
        Updated velocity vector
    n_encounters : int
        Number of encounters in this timestep
    """
    # Current speed
    speed = np.linalg.norm(velocity_vector)
    
    # Distance traveled in this time step
    dx = speed * dt
    
    # Volume of cylinder slice
    dV = np.pi * R**2 * dx
    
    # Expected number of encounters
    lambda_encounters = n * dV
    
    # Generate actual number of encounters (Poisson process)
    n_encounters = np.random.poisson(lambda_encounters)
    
    # Process each encounter
    new_velocity = velocity_vector.copy()
    
    # for _ in range(n_encounters):
    # Generate encounter geometry
    b, phi = generate_encounter(R,n_encounters)
        
    # Calculate velocity change
    for i in range(n_encounters):
        v_rel = speed  # Assuming field particles at rest
        delta_v = impulse_approximation(v_rel, b[i], M, G)
        
        # Apply velocity change
        new_velocity = apply_velocity_change(new_velocity, delta_v, phi[i])
    
    return new_velocity, n_encounters



def run_relaxation_experiment(m, M, v0, n, R, G, dt, max_time, verbose=True):
    """
    Run the complete two-body relaxation experiment.
    
    Parameters:
    -----------
    m : float
        Test particle mass
    M : float
        Field particle mass
    v0 : float
        Initial speed
    n : float
        Number density of field particles
    R : float
        Cylinder radius
    G : float
        Gravitational constant
    dt : float
        Time step
    max_time : float
        Maximum simulation time
    verbose : bool
        Print progress updates
        
    Returns:
    --------
    results : dict
        Dictionary containing time series data and final statistics
    """
    if verbose:
        print("Starting two-body relaxation experiment...")
        print(f"Initial conditions: v0={v0}, n={n}, M={M}, R={R}")
    
    # Initialize
    t = 0
    velocity_vector = np.array([0, 0, v0])  # Initial velocity in z-direction
    
    # Storage arrays
    times = []
    speeds = []
    velocity_vectors = []
    delta_v_cumulative = []
    encounter_counts = []
    
    # Calculate theoretical prediction
    t_theory = theoretical_relaxation_time(v0, n, M, m, G, R)
    if verbose:
        print(f"Theoretical relaxation time: {t_theory:.2f}")
    
    start_time = time.time()
    
    # Main evolution loop
    while t < max_time:
        # Store current state
        times.append(t)
        current_speed = np.linalg.norm(velocity_vector)
        speeds.append(current_speed)
        velocity_vectors.append(velocity_vector.copy())
        
        # Calculate cumulative change from initial velocity
        delta_v_cum = abs(current_speed - v0)
        delta_v_cumulative.append(delta_v_cum)
        
        # Check stopping condition
        if delta_v_cum >= v0:
            if verbose:
                print(f"Relaxation achieved at t = {t:.2f}")
                print(f"Final speed: {current_speed:.3f} (initial: {v0:.3f})")
            break
        
        # Evolve one time step
        velocity_vector, n_enc = run_single_timestep(velocity_vector, dt, n, R, M, G)
        encounter_counts.append(n_enc)
        
        t += dt
        
        # Progress update
        if verbose and len(times) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"t = {t:.1f}, speed = {current_speed:.3f}, "
                  f"Δv_cum = {delta_v_cum:.3f}, encounters = {n_enc}, "
                  f"elapsed = {elapsed:.1f}s")
    
    # Compile results
    results = {
        'times': np.array(times),
        'speeds': np.array(speeds),
        'velocity_vectors': np.array(velocity_vectors),
        'delta_v_cumulative': np.array(delta_v_cumulative),
        'encounter_counts': np.array(encounter_counts),
        'relaxation_time_numerical': t if delta_v_cumulative[-1] >= v0 else None,
        'relaxation_time_theoretical': t_theory,
        'final_speed': np.linalg.norm(velocity_vector),
        'initial_speed': v0,
        'total_encounters': np.sum(encounter_counts),
        'parameters': {
            'm': m, 'M': M, 'v0': v0, 'n': n, 'R': R, 'G': G, 'dt': dt
        }
    }
    
    if verbose:
        print(f"\nExperiment completed!")
        print(f"Total simulation time: {time.time() - start_time:.1f} seconds")
        print(f"Total encounters: {results['total_encounters']}")
        
    return results


def plot_results(results, save_filename=None):
    """
    Create comprehensive plots of the relaxation experiment results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_relaxation_experiment
    save_filename : str, optional
        Filename to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    times = results['times']
    speeds = results['speeds']
    delta_v_cum = results['delta_v_cumulative']
    v0 = results['initial_speed']
    t_theory = results['relaxation_time_theoretical']
    t_numerical = results['relaxation_time_numerical']
    
    # Speed evolution
    axes[0, 0].plot(times, speeds, 'b-', linewidth=1, alpha=0.8)
    axes[0, 0].axhline(v0, color='r', linestyle='--', label=f'Initial speed: {v0:.1f}')
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Speed')
    axes[0, 0].set_title('Speed Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative velocity change
    axes[0, 1].plot(times, delta_v_cum, 'g-', linewidth=1)
    axes[0, 1].axhline(v0, color='r', linestyle='--', label=f'Target: {v0:.1f}')
    if t_theory is not None:
        axes[0, 1].axvline(t_theory, color='orange', linestyle='--', 
                          label=f'Theory: {t_theory:.1f}')
    if t_numerical is not None:
        axes[0, 1].axvline(t_numerical, color='purple', linestyle='-', 
                          label=f'Numerical: {t_numerical:.1f}')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('|Δv| cumulative')
    axes[0, 1].set_title('Cumulative Velocity Change')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Encounter rate
    if len(results['encounter_counts']) > 0:
        axes[1, 0].plot(times[:-1], results['encounter_counts'], 'r-', alpha=0.6)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Encounters per timestep')
        axes[1, 0].set_title('Encounter Rate')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity components
    vel_vectors = results['velocity_vectors']
    axes[1, 1].plot(times, vel_vectors[:, 0], label='vx', alpha=0.8)
    axes[1, 1].plot(times, vel_vectors[:, 1], label='vy', alpha=0.8)
    axes[1, 1].plot(times, vel_vectors[:, 2], label='vz', alpha=0.8)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Velocity components')
    axes[1, 1].set_title('Velocity Vector Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_filename}")
    
    plt.show()


def parameter_study(base_params, vary_param, param_values, max_time=1000):
    """
    Run parameter study varying one parameter while keeping others fixed.
    
    Parameters:
    -----------
    base_params : dict
        Base parameter set
    vary_param : str
        Name of parameter to vary
    param_values : array
        Values of the parameter to test
    max_time : float
        Maximum simulation time for each run
        
    Returns:
    --------
    study_results : dict
        Results for each parameter value
    """
    print(f"Running parameter study varying {vary_param}...")
    
    study_results = {
        'param_values': param_values,
        'relaxation_times_numerical': [],
        'relaxation_times_theoretical': [],
        'final_speeds': []
    }
    
    for i, param_val in enumerate(param_values):
        print(f"\n--- Run {i+1}/{len(param_values)}: {vary_param} = {param_val} ---")
        
        # Update parameters
        params = base_params.copy()
        params[vary_param] = param_val
        
        # Run experiment
        results = run_relaxation_experiment(
            params['m'], params['M'], params['v0'], params['n'], 
            params['R'], params['G'], params['dt'], max_time, verbose=False
        )
        
        # Store results
        study_results['relaxation_times_numerical'].append(results['relaxation_time_numerical'])
        study_results['relaxation_times_theoretical'].append(results['relaxation_time_theoretical'])
        study_results['final_speeds'].append(results['final_speed'])
        
        print(f"Relaxation time: {results['relaxation_time_numerical']:.2f} "
              f"(theory: {results['relaxation_time_theoretical']:.2f})")
    
    return study_results


def main_experiment(m=1.0, M=1.0, v0=10.0, n=1e-3, R=100.0, G=1.0, dt=0.01, max_time=1000.0):
    """
    Main function to run the two-body relaxation experiment with specified parameters.
    
    This is the primary interface for running the experiment from a notebook.
    
    Parameters:
    -----------
    m : float, default=1.0
        Test particle mass
    M : float, default=1.0  
        Field particle mass
    v0 : float, default=10.0
        Initial speed
    n : float, default=1e-3
        Number density of field particles
    R : float, default=100.0
        Cylinder radius
    G : float, default=1.0
        Gravitational constant
    dt : float, default=0.01
        Time step
    max_time : float, default=1000.0
        Maximum simulation time
        
    Returns:
    --------
    results : dict
        Complete results dictionary
    """
    print("="*60)
    print("TWO-BODY GRAVITATIONAL RELAXATION EXPERIMENT")
    print("="*60)
    
    # Run the experiment
    results = run_relaxation_experiment(m, M, v0, n, R, G, dt, max_time)
    
    # Create plots
    plot_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Theoretical relaxation time: {results['relaxation_time_theoretical']:.2f}")
    print(f"Numerical relaxation time:   {results['relaxation_time_numerical']}")
    print(f"Ratio (numerical/theory):    {results['relaxation_time_numerical']/results['relaxation_time_theoretical']:.2f}" 
          if results['relaxation_time_numerical'] else "N/A")
    print(f"Initial speed:               {results['initial_speed']:.2f}")
    print(f"Final speed:                 {results['final_speed']:.2f}")
    print(f"Total encounters:            {results['total_encounters']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = main_experiment(
        m=1.0,      # Test particle mass
        M=1.0,      # Field particle mass  
        v0=10.0,    # Initial velocity
        n=5e-4,     # Number density
        R=100.0,    # Cylinder radius
        G=1.0,      # Gravitational constant
        dt=0.01,    # Time step
        max_time=500.0  # Maximum time
    )
