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

from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import numpy as np


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



# === helpers.py additions ===
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

# Cached HELIO4CAST ICME catalog
_HELIO_URL = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
_CME_CATALOG = None

def load_cme_catalog():
    """Download & cache the HELIO4CAST ICME catalog with parsed datetimes."""
    global _CME_CATALOG
    if _CME_CATALOG is not None:
        return _CME_CATALOG

    df = pd.read_csv(_HELIO_URL)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    for col in ["icme_start_time", "mo_start_time", "mo_end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["sc_insitu"] = df["sc_insitu"].astype(str).str.strip()
    _CME_CATALOG = df
    return _CME_CATALOG

def get_cme_times_datetime(t_start, t_end, spacecraft="Wind"):
    """
    Return a list of ICME start times between t_start and t_end for a given spacecraft.
    """
    df = load_cme_catalog()
    # Ensure datetime comparison is valid
    mask = (
        (df["icme_start_time"] >= pd.to_datetime(t_start)) &
        (df["icme_start_time"] <= pd.to_datetime(t_end)) &
        (df["sc_insitu"].str.lower() == spacecraft.lower())
    )
    return df.loc[mask, "icme_start_time"].to_list()

def carr_days_to_datetime(x_days, cr_start):
    """Vectorized: days since CR start -> UTC datetime."""
    t0 = pd.to_datetime(cr_start, utc=True)
    return t0 + pd.to_timedelta(np.asarray(x_days, dtype=float), unit="D")


def convert_axis_days_to_datetime(ax, cr_start):
    """
    Convert all line x-data on 'ax' from days-since-CR to UTC datetimes
    and set date formatting. Call *after* any day-based computations (e.g., RMSE).
    """
    for ln in ax.get_lines():
        x = ln.get_xdata()
        if len(x) and np.issubdtype(np.asarray(x).dtype, np.number):
            ln.set_xdata(carr_days_to_datetime(x, cr_start))

    ax.relim(); ax.autoscale_view()
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def shade_interval_datetime(ax, t0, t1, *, color="0.85", alpha=0.25, label=None, zorder=0):
    """Generic datetime shading (e.g., re-drawing the prograde window)."""
    t0 = pd.to_datetime(t0, utc=True)
    t1 = pd.to_datetime(t1, utc=True)
    ax.axvspan(t0, t1, color=color, alpha=alpha, label=label, zorder=zorder)

def find_first_divergence_window(x_days_model, v_model,
                                 x_days_l1,    v_l1,
                                 start_day=0.0, thresh=20.0, min_hours=6.0):
    """
    Return (t_start_days, t_end_days) of the first sustained divergence window
    where |model - L1| > thresh (km/s) for at least min_hours.
    Times are in DAYS since CR start.  None if not found.
    """
    import numpy as np
    import pandas as pd

    # L1 -> model grid
    l1_on_model = np.interp(x_days_model, x_days_l1, v_l1, left=np.nan, right=np.nan)
    diff = np.abs(v_model - l1_on_model)

    # only after start_day (beginning of the plot)
    m = np.isfinite(diff) & (x_days_model >= float(start_day))
    if not np.any(m):
        return None

    xd = x_days_model[m]
    dd = diff[m]

    # moving average to enforce persistence
    if len(xd) < 2:
        return None
    dt_days = float(np.median(np.diff(xd)))
    win = max(1, int(np.ceil((min_hours/24.0) / max(dt_days, 1e-9))))
    dd_ma = pd.Series(dd).rolling(win, min_periods=1).mean().to_numpy()

    # first time above threshold
    above = dd_ma > float(thresh)
    if not np.any(above):
        return None
    i0 = np.argmax(above)                           # first True
    # walk until it drops back below threshold
    i1 = i0
    while i1 < len(xd) and above[i1]:
        i1 += 1
    return float(xd[i0]), float(xd[min(i1, len(xd)-1)])



def shade_cmes_datetime(ax, t_min, t_max, *, spacecraft="Wind", catalog=None):
    """
    Shade CME intervals on a datetime x-axis using HELIO4CAST catalog.
    Only intervals overlapping [t_min, t_max] are drawn.
    """
    if catalog is None:
        catalog = load_cme_catalog()

    sub = catalog[catalog["sc_insitu"].str.contains(spacecraft, case=False, na=False)]
    labeled = False
    for s, e in sub[["icme_start_time", "mo_end_time"]].itertuples(index=False, name=None):
        if pd.isna(s) or pd.isna(e):
            continue
        if (s <= t_max) and (e >= t_min):
            ax.axvspan(s, e, color="orange", alpha=0.20,
                       label=("L1 CME" if not labeled else None), zorder=0)
            labeled = True

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
    Generate accelerated PSP velocity profile from MAS model,
    plot original and adjusted PSP points vs all MAS profiles,
    and save the plot.
    """
    import os
    import matplotlib.pyplot as plt

    df = get_prograde_df(encounter_number)

    # Load MAS model data
    models = pickle.load(open('../data/the_models_2025-05-01', 'rb'))
    radii_ro = models[0][0].to_value(u.solRad)                    # Radius grid
    vel_profiles = np.array([v.to_value(u.km/u.s) for v in models[0][2]])  # Velocity curves

    boosted = []
    r_vals = df['radius'].values
    v_vals = df['Vr'].values

    for v0, r0 in zip(v_vals, r_vals):
        r_idx = np.argmin(np.abs(radii_ro - r0))
        v_col = vel_profiles[:, r_idx]
        p_idx = np.argmin(np.abs(v_col - v0))
        v30  = vel_profiles[p_idx, np.argmin(np.abs(radii_ro - 30))]
        boosted.append(v30)

    # Interpolate to fill LON_GRID
    interp = interp1d(df['longitude'], boosted, kind='linear', bounds_error=False, fill_value=np.nan)
    vr_acc = interp(_LON_GRID)
    merged_vals = np.where(np.isnan(vr_acc), v_mas.value, vr_acc)

    # Wraparound fix
    merged_vals = _patch_wraparound(df, merged_vals, v_mas)

    # --------- Plotting ---------
    os.makedirs("huxt_outputs/accel_profiles", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot all MAS model velocity profiles
    for profile in vel_profiles:
        ax.plot(radii_ro, profile, lw=1, alpha=0.6)

    # Overlay original PSP Vr points
    ax.scatter(r_vals, v_vals, color='blue', label='Original PSP Vr', zorder=10, s=20)

    # Overlay adjusted points at 30 R☉
    ax.scatter([30]*len(boosted), boosted, color='maroon', label='Adjusted Vr at 30 R($\odot$)',              zorder=11, s=20)

    # Vertical line at 30 R☉
    ax.axvline(x=30, color='black', linestyle='--', lw=1.2)

    ax.set_xlabel("Radius (R$\odot$)")
    ax.set_ylabel("Vr (km/s)")
    ax.set_title(f"Accelerated PSP Points – Encounter {encounter_number}")
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 800)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # Save to disk
    fig.savefig(f"huxt_outputs/accel_profiles/enc{encounter_number:02d}_accel_profile.png", dpi=300)
    plt.close(fig)

    return merged_vals * u.km/u.s


   
import numpy as np

def shade_prograde_region(ax, diff_thresh=10, span_days=7, min_day=10):
    """
    Shade from the first divergence after `min_day` to the last divergence
    within `span_days` of that point.
    """
    # grab your two curves
    vacc = next(l for l in ax.get_lines() if l.get_label().startswith('Vaccel'))
    vmas = next(l for l in ax.get_lines() if l.get_label().startswith('Vmas'))

    # get raw arrays (strip units if needed)
    x  = vacc.get_xdata();  y1 = vacc.get_ydata();  y2 = vmas.get_ydata()
    if hasattr(x,  'unit'): x  = x.value
    if hasattr(y1, 'unit'): y1 = y1.value
    if hasattr(y2, 'unit'): y2 = y2.value

    # build divergence mask
    mask = np.abs(y1 - y2) > diff_thresh
    # **ignore anything before min_day**
    mask &= (x >= min_day)

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None, None

    # first True after min_day → x_min
    x_min = x[idxs[0]]

    # look out span_days → last True in that window → x_max
    cutoff      = x_min + span_days
    within_win  = idxs[x[idxs] <= cutoff]
    x_max = x[within_win[-1]] if within_win.size else x_min

    # shade and return
    ax.axvspan(x_min, x_max, color='gray', alpha=0.3)
    return x_min, x_max


def compute_region_rmse(ax, x_min, x_max):
    """
    Compute RMSE of Vaccel, Vmerged, and Vmas vs L1 obs, but only for the
    time window [x_min, x_max], and only where the L1 data is finite.
    """
    # 1) grab lines
    vacc_line    = next(l for l in ax.get_lines() if l.get_label().startswith('Vaccel'))
    vmerged_line = next(l for l in ax.get_lines() if l.get_label().startswith('Vmerged'))
    vmas_line    = next(l for l in ax.get_lines() if l.get_label().startswith('Vmas'))
    l1_line      = next(l for l in ax.get_lines() if l.get_label().startswith('L1'))

    # 2) pull out arrays
    x_mod = vacc_line.get_xdata();   y_acc = vacc_line.get_ydata()
    y_mer = vmerged_line.get_ydata()
    y_mas = vmas_line.get_ydata()
    x_l1  = l1_line.get_xdata();     y_l1  = l1_line.get_ydata()

    # 3) strip units if needed
    if hasattr(x_mod, 'unit'): x_mod = x_mod.value
    if hasattr(y_acc, 'unit'): y_acc = y_acc.value
    if hasattr(y_mer, 'unit'): y_mer = y_mer.value
    if hasattr(y_mas, 'unit'): y_mas = y_mas.value
    if hasattr(x_l1,  'unit'): x_l1  = x_l1.value
    if hasattr(y_l1,  'unit'): y_l1  = y_l1.value

    # 4) mask L1 to [x_min, x_max] and finite
    in_window = (x_l1 >= x_min) & (x_l1 <= x_max)
    finite_obs = np.isfinite(y_l1)
    mask_l1 = in_window & finite_obs

    t_l1 = x_l1[mask_l1]
    v_l1 = y_l1[mask_l1]

    if t_l1.size == 0:
        # nothing to compare
        return np.nan, np.nan, np.nan

    # 5) interpolate each model onto the L1 times that survive
    acc_on_l1 = np.interp(t_l1, x_mod, y_acc)
    mer_on_l1 = np.interp(t_l1, x_mod, y_mer)
    mas_on_l1 = np.interp(t_l1, x_mod, y_mas)

    # 6) compute rmse
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    return (
        rmse(acc_on_l1, v_l1),
        rmse(mer_on_l1, v_l1),
        rmse(mas_on_l1, v_l1),
    )


def plot_alignment(df, df_full, title='PSP–Earth Alignment'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy import units as u

    RS_to_AU = 1 / 215.0

    # --- PSP midpoint & full segment ---
    pro_lons = df['longitude']
    pro_rs   = df['radius'] * RS_to_AU

    # full prograde arc
    pro_arc_x = pro_rs * np.cos(np.deg2rad(pro_lons))
    pro_arc_y = pro_rs * np.sin(np.deg2rad(pro_lons))

    # midpoint (PSP)
    mid_idx = len(df) // 2
    mid_lon = pro_lons.iloc[mid_idx]
    mid_r   = pro_rs.iloc[mid_idx]
    pro_x   = mid_r * np.cos(np.deg2rad(mid_lon))
    pro_y   = mid_r * np.sin(np.deg2rad(mid_lon))

    # PSP midpoint date
    psp_date = pd.to_datetime(df['time'].iloc[mid_idx])

    # --- Earth (L1) ---   
    lon_col = [c for c in df_full.columns if 'Lon' in c and 'L1' in c][0]
    r_col   = [c for c in df_full.columns if c.startswith('R') and 'L1' in c][0]

    # Earth longitude/radius at PSP index
    earth_lon = df_full.loc[df.index[mid_idx], lon_col]
    earth_r   = df_full.loc[df.index[mid_idx], r_col] * RS_to_AU

    # --- Propagate solar wind travel time (~400 km/s) ---
    distance_au = abs(earth_r - mid_r)
    distance_km = (distance_au * u.AU).to(u.km).value
    travel_time_sec = distance_km / 400.0    # speed = 400 km/s

    earth_date = psp_date + pd.to_timedelta(travel_time_sec, unit='s')

    # Earth arc points
    arc_center_x = earth_r * np.cos(np.deg2rad(earth_lon))
    arc_center_y = earth_r * np.sin(np.deg2rad(earth_lon))
    arc_lons = (earth_lon + np.linspace(-45, 45, 200)) % 360
    arc_r    = np.full_like(arc_lons, earth_r)
    arc_x    = arc_r * np.cos(np.deg2rad(arc_lons))
    arc_y    = arc_r * np.sin(np.deg2rad(arc_lons))

    # --- Alignment angle ---
    alignment_angle = earth_lon - mid_lon

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(0, 0, 'o', color='gold', label='Sun', markersize=8, zorder=4)

    # full PSP segment
    ax.plot(pro_arc_x, pro_arc_y, color='red', lw=1.2, alpha=0.8, label='Prograde segment')

    # PSP midpoint (with date)
    ax.plot(pro_x, pro_y, 'o', color='red',
            label=f"PSP midpoint ({psp_date:%Y/%m/%d %H:00})", zorder=5)

    # Earth point + arc (with date)
    ax.plot(arc_center_x, arc_center_y, 'o', color='blue',
            label=f"Earth (L1) ({earth_date:%Y/%m/%d %H:00})", zorder=5)
    ax.plot(arc_x, arc_y, color='blue', lw=1.2, alpha=0.8, label='Earth arc')

    # alignment lines
    ax.plot([0, pro_x], [0, pro_y], 'r--', alpha=0.6)
    ax.plot([0, arc_center_x], [0, arc_center_y], 'b--', alpha=0.6)

    # formatting
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_title(title)
    ax.legend(loc='lower left')

    # alignment angle text at top right
    ax.text(0.95, 0.95, f'Alignment angle: {alignment_angle:.1f}°',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig, ax, alignment_angle

# def plot_alignment(df, cr_start, avg_speed_kms, title='PSP–Earth Alignment'):
#     """
#     Plot PSP prograde segment and Earth's ¼‑Carrington rotation arc
#     in the Carrington frame, then compute & display the alignment angle.
#     `df` must be the prograde-only DataFrame from hp.get_prograde_df().
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # constants
#     AU_km     = 1.496e8           
#     RS_to_AU  = 1/215.0            
#     carr_rate = 360/27.2753        # deg per day

#     # psp prograde segment in AU
#     pro_lons = df['longitude'] % 360
#     pro_rs   = df['radius'] * RS_to_AU
#     pro_x    = pro_rs * np.cos(np.deg2rad(pro_lons))
#     pro_y    = pro_rs * np.sin(np.deg2rad(pro_lons))

#     # compute Earth's lagged Carrington longitude 
#     mean_r   = pro_rs.mean()
#     lag_sec  = (1.0 - mean_r) * AU_km / avg_speed_kms
#     lag_days = lag_sec / 86400
#     earth_lon = (lag_days * carr_rate) % 360

#     # build Earth's ¼‑rotation arc (90°)
#     arc_lons = (earth_lon + np.linspace(0, 90, 200)) % 360
#     arc_x    = np.cos(np.deg2rad(arc_lons))
#     arc_y    = np.sin(np.deg2rad(arc_lons))

#     # alignment angle between mean PSP lon & earth_lon 
#     mean_psp_lon     = pro_lons.mean()
#     delta            = (earth_lon - mean_psp_lon + 360) % 360
#     alignment_angle  = delta if delta <= 180 else 360 - delta

#     # plotting
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.plot(0, 0,                      'o', color='gold', label='Sun')
#     ax.plot(pro_x, pro_y,             lw=2, color='red',  label='PSP prograde')
#     ax.plot(arc_x, arc_y,             lw=2, color='blue', label='Earth ¼‑Carr. rot.')
#     # dashed radial lines (optional)
#     ax.plot([0, pro_x.iloc[0]],       [0, pro_y.iloc[0]],       'r--', alpha=0.5)
#     ax.plot([0, np.cos(np.deg2rad(earth_lon))],
#             [0, np.sin(np.deg2rad(earth_lon))],                  'b--', alpha=0.5)

#     ax.set_aspect('equal')
#     ax.margins(0.27)
#     # ax.set_xlim(-0.5, 1.5)
#     # ax.set_ylim(-0.5, 1.5)
#     ax.set_xlabel('X (AU)')
#     ax.set_ylabel('Y (AU)')
#     ax.set_title(title)
#     ax.legend(loc='upper right')
#     ax.text(0.95, 0.05,
#             f'Alignment angle: {alignment_angle:.1f}°',
#             transform=ax.transAxes,
#             ha='right',va='bottom',
#             fontsize=10, fontweight='bold')

#     plt.tight_layout()
#     return fig, ax, alignment_angle




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
