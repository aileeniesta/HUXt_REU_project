import datetime
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d 
import sunpy.coordinates

def gen_dt_arr(dt_init,dt_final,cadence_days=1) :
    """
    'Generate Datetime Array'
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += datetime.timedelta(days=cadence_days)
    return np.array(dt_list)

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])
def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.datetime.utcfromtimestamp(ut) for ut in ut_arr])
def interp_trajectory(dt_in, trajectory_in, dt_out) :
    trajectory_in.representation_type="spherical"
    radius_in = trajectory_in.radius.to("R_sun").value
    lon_in = trajectory_in.lon.to("deg").value
    lat_in = trajectory_in.lat.to("deg").value
    disconts = np.where(np.abs(np.diff(lon_in)) > 60)[0]
    nan_inds = []
    for discont in disconts: 
        nan_inds.append(discont)
        nan_inds.append(discont+1)
    radius_in[nan_inds] = np.nan
    lon_in[nan_inds] = np.nan
    lat_in[nan_inds] = np.nan
    return SkyCoord(
        radius=interp1d(datetime2unix(dt_in),radius_in,bounds_error=False)(datetime2unix(dt_out))*u.R_sun,
        lon=interp1d(datetime2unix(dt_in),lon_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,
        lat=interp1d(datetime2unix(dt_in),lat_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,        
        representation_type="spherical",
        frame=sunpy.coordinates.HeliographicCarrington,
        obstime=dt_out)
@u.quantity_input
def delta_long(r:u.R_sun,
               r_inner=2.5*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if skycoord.representation_type != "spherical" :
        skycoord.representation_type="spherical"
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )

def roll_to_180(lons) : 
    return ((lons + 180) % 360)-180

from pathlib import Path
from astropy.time import Time     

def get_prograde_df(enc):
    """
    Read raw encounter file ../data/E##.csv, keep only the pro‑grade
    portion, and return a tidy DataFrame with columns:
        time · longitude · radius · latitude · Vr
    """
    df = pd.read_csv(f"../data/E{enc:02d}.csv")
    df.columns = [c.replace('-', '_').replace('–','_').strip() for c in df.columns]
    df['Times'] = pd.to_datetime(df['Times'], errors='coerce')
    df = df.dropna(subset=['Times', 'Vpr_Parker', 'Lon_Parker'])

    # flag pro‑grade motion (Δlon in (0, 180)°)
    df['prograde'] = (df['Lon_Parker'].diff() > 0) & (df['Lon_Parker'].diff() < 180)
    prog_df = df.loc[df['prograde']].copy()

    # tidy columns
    result = prog_df[['Times','Lon_Parker','R_Parker','Lat_Parker','Vpr_Parker']]\
               .rename(columns={'Times':'time',
                                'Lon_Parker':'longitude',
                                'R_Parker':'radius',
                                'Lat_Parker':'latitude',
                                'Vpr_Parker':'Vr'})
    return result.reset_index(drop=True)



from functools        import lru_cache
from pathlib          import Path

_CR0     = Time('1853-11-09 12:00:00', format='iso', scale='utc')
_CR_LEN  = 27.2753         # days

def carrington_info(timestamp):
    """Return (cr_number, cr_start_datetime) for a UTC timestamp."""
    cr_num   = int(((Time(timestamp) - _CR0).to('day').value) / _CR_LEN) + 1
    cr_start = _CR0 + (cr_num - 1)*_CR_LEN*u.day
    return cr_num, cr_start.to_datetime()

def days_since(series, start_dt):
    """Return array of days since start_dt."""
    series = np.array(series, dtype='datetime64[ns]')
    return (series - np.datetime64(start_dt)).astype('timedelta64[s]').astype(float)/86400.0

def _rename_columns(df):
    """Make column names uniform."""
    return df.rename(columns={
        'Times':'time', 'Lon_Parker':'longitude', 'Vr_Parker':'Vr'
    })



import pickle

_LON_GRID = np.arange(0, 360, 1)        

def _patch_wraparound(df, merged_vals, v_mas, gap_threshold=15):
    lons  = np.sort(np.mod(df['longitude'].values, 360))
    diffs = np.diff(np.concatenate([lons, lons[:1] + 360]))
    idx   = np.argmax(diffs)
    if diffs[idx] <= gap_threshold:
        return merged_vals                          # nothing to fix

    g0, g1 = lons[idx], lons[(idx + 1) % lons.size]
    lon_grid = np.arange(0, 360, 1)
    mask = ((lon_grid > g0) & (lon_grid < g1)) if g0 < g1 else \
           ((lon_grid > g0) | (lon_grid < g1))

    merged_vals[mask] = v_mas.value[mask]
    return merged_vals


def get_v_merged(encounter_number: int, v_mas: u.Quantity) -> u.Quantity:
    """
    PSP Vr interpolated onto 1° grid, MAS fills the gaps.
    Now auto‑patches any wrap‑around longitude gap >15°.
    """
    df = get_prograde_df(encounter_number)

    # --- linear interpolation of PSP Vr onto the 1° grid -------------
    interp            = interp1d(df['longitude'], df['Vr'],
                                 kind='linear', bounds_error=False,
                                 fill_value=np.nan)
    vr_psp_on_grid    = interp(_LON_GRID)                 # <-- keep this
    merged_vals       = np.where(np.isnan(vr_psp_on_grid),
                                 v_mas.value,
                                 vr_psp_on_grid)

    merged_vals = _patch_wraparound(df, merged_vals, v_mas)

    return merged_vals * u.km/u.s



def get_v_merged_accel(encounter_number: int, v_mas: u.Quantity) -> u.Quantity:
    """
    Same idea, but starts from the accelerated PSP points.
    """
    df = get_prograde_df(encounter_number)

    # ---------- create the accelerated list (original code) -----
    models          = pickle.load(open('../data/the_models_2025-05-01', 'rb'))
    radii_ro        = models[0][0].to_value(u.solRad)          # radii grid#    
    vel_profiles    = np.array([v.to_value(u.km/u.s) for v in models[0][2]])

    boosted = []
    for v0, r0 in zip(df['Vr'], df['radius']):
        r_idx = np.argmin(np.abs(radii_ro - r0))
        v_col = vel_profiles[:, r_idx]
        p_idx = np.argmin(np.abs(v_col - v0))
        v30   = vel_profiles[p_idx, np.argmin(np.abs(radii_ro - 30))]
        boosted.append(v30)

    interp   = interp1d(df['longitude'], boosted,
                        kind='linear', bounds_error=False,
                        fill_value=np.nan)
    vr_acc   = interp(_LON_GRID)
    merged_vals = np.where(np.isnan(vr_acc), v_mas.value, vr_acc)

    # ---------- wrap‑around patch --------------------------------
    merged_vals = _patch_wraparound(df, merged_vals, v_mas)

    return merged_vals * u.km/u.s


def shade_divergent_regions(ax, diff_thresh=10):
    """
    Shade all regions where Vaccel and Vmas differ by more than diff_thresh (km/s).
    Works even when lines cross over or rejoin.
    """
    vacc = next(l for l in ax.get_lines() if l.get_label().startswith('Vaccel'))
    vmas = next(l for l in ax.get_lines() if l.get_label().startswith('Vmas'))

    x = vacc.get_xdata()
    y1 = vacc.get_ydata()
    y2 = vmas.get_ydata()

    if hasattr(x, 'unit'): x = x.value
    if hasattr(y1, 'unit'): y1 = y1.value
    if hasattr(y2, 'unit'): y2 = y2.value

    diff = np.abs(y1 - y2)
    mask = diff > diff_thresh

    # Find continuous chunks using changes in mask
    in_region = False
    for i in range(1, len(mask)):
        if mask[i] and not in_region:
            # Start of new region
            start = x[i]
            in_region = True
        elif not mask[i] and in_region:
            # End of region
            end = x[i]
            ax.axvspan(start, end, color='gray', alpha=0.1)
            in_region = False

    # If still in a region at end of array
    if in_region:
        ax.axvspan(start, x[-1], color='gray', alpha=0.1)





def plot_alignment(df, cr_start, avg_speed_kms, title='PSP–Earth Alignment'):

    """
    Plot PSP, Sun, and Earth positions in Carrington frame with solar wind lag.
    Converts PSP radius from solar radii to AU.
    
    
    Returns:
    - fig, ax, alignment_angle_deg
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Constants
    AU_km = 1.496e8
    RS_to_AU = 1 / 215.0  # solar radii to AU
    carr_rate = 360 / 27.2753  # deg/day

    # PSP location (in AU)
    psp_lon = np.mean(df['longitude']) % 360
    psp_r = np.mean(df['radius']) * RS_to_AU

    # Lag time from PSP to Earth
    lag_sec = (1.0 - psp_r) * AU_km / avg_speed_kms
    lag_days = lag_sec / 86400
    earth_lon = (lag_days * carr_rate) % 360

    # Convert to Cartesian
    psp_x = psp_r * np.cos(np.deg2rad(psp_lon))
    psp_y = psp_r * np.sin(np.deg2rad(psp_lon))
    earth_x = 1.0 * np.cos(np.deg2rad(earth_lon))
    earth_y = 1.0 * np.sin(np.deg2rad(earth_lon))

    # Alignment angle
    delta_lon = (earth_lon - psp_lon + 360) % 360
    alignment_angle_deg = delta_lon if delta_lon <= 180 else 360 - delta_lon

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(0, 0, 'o', color='gold', label='Sun')
    ax.plot(psp_x, psp_y, 'o', color='red', label='PSP')
    ax.plot(earth_x, earth_y, 'o', color='blue', label='Earth (lagged)')

    ax.plot([0, psp_x], [0, psp_y], 'r--', alpha=0.5)
    ax.plot([0, earth_x], [0, earth_y], 'b--', alpha=0.5)

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.text(0, 1, f'Alignment angle: {alignment_angle_deg:.1f}°', fontsize=10,                               fontweight='bold')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    plt.tight_layout()

    return fig, ax, alignment_angle_deg



def delta_long(r_inner, r_outer, vsw):
    """
    Simple ballistic‐corotation longitude lag (deg) between two radii.
    Assumes rigid rotation at 360° / 27.2753 d.
    """
    tof   = (r_outer-r_inner).to(u.km) / vsw    # time‑of‑flight
    omega = 360.0/_CR_LEN * u.deg/u.day
    return (omega*tof).to_value(u.deg)


def get_time_intervals_from_mask(mask, time_array):
    """
    Convert a boolean mask and corresponding time array into a list of (start, end) intervals.
    Each interval corresponds to a contiguous True region in the mask.
    """
    intervals = []
    in_interval = False
    for i in range(len(mask)):
        if mask[i] and not in_interval:
            start = time_array[i]
            in_interval = True
        elif not mask[i] and in_interval:
            end = time_array[i - 1]
            intervals.append((start, end))
            in_interval = False
    if in_interval:
        intervals.append((start, time_array[-1]))
    return intervals


def rmse(a, b, times=None, intervals=None):
    """
    Compute RMSE between arrays `a` and `b` ONLY within specified `intervals`.

    Parameters:
    - a, b: arrays to compare
    - times: array of times (same length as `a` and `b`)
    - intervals: list of (start, end) time ranges

    Returns:
    - RMSE value or np.nan if no valid data
    """
    import numpy as np
    a = np.array(a)
    b = np.array(b)

    if times is None or intervals is None:
        valid = ~np.isnan(a) & ~np.isnan(b)
    else:
        times = np.array(times)
        mask = np.zeros_like(times, dtype=bool)
        for start, end in intervals:
            mask |= (times >= start) & (times <= end)
        valid = mask & ~np.isnan(a) & ~np.isnan(b)

    return np.sqrt(np.mean((a[valid] - b[valid])**2)) if valid.any() else np.nan
