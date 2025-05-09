import numpy as np
import xarray as xr
import rasterio
import os
import matplotlib.pyplot as plt

def terrain_correction(elevation, integration_radius, latP, longP, HP=None, G=6.67259e-11, rho=2670):
    """
    Compute terrain correction at a point based on a DEM.

    Parameters:
        elevation (xr.DataArray): Elevation grid with 'latitude' and 'longitude'.
        integration_radius (float or None): Integration radius in meters (use None for full area).
        latP (float): Latitude of the station in decimal degrees.
        longP (float): Longitude of the station in decimal degrees.
        HP (float or None): Elevation of the station. If None, it will be extracted from `elevation`.
        G (float): Gravitational constant (default: 6.67259e-11).
        rho (float): Rock density in kg/m³ (default: 2670).

    Returns:
        float: Total terrain correction in mGal.
    """
    if HP is None:
        HP = elevation.sel(latitude=latP, longitude=longP, method="nearest").item()

    # Coordinate grids
    lon_grid, lat_grid = np.meshgrid(elevation["longitude"].values, elevation["latitude"].values)

    # Deltas in meters (1 arcsecond ≈ 30 m)
    delta_lat = (lat_grid - latP) * 3600 * 30
    delta_lon = (lon_grid - longP) * 3600 * 30

    # Horizontal distance
    distance = np.sqrt(delta_lat**2 + delta_lon**2)

    # Resolution
    res_long = np.mean(np.abs(np.diff(elevation.coords['longitude'].values)))
    res_lat = np.mean(np.abs(np.diff(elevation.coords['latitude'].values)))
    res = np.median([res_long, res_lat])
    res_m = round((res * 30)*3600)
    res_m2 = res_m**2

    # Constants in mGal
    cte1 = (G * rho / 2) * res_m2 * 1e5
    cte2 = (3 * G * rho / 8) * res_m2 * 1e5

    # Elevation difference
    delta_h = elevation - HP

    # Mask by integration radius
    if integration_radius is not None:
        mask = distance <= integration_radius
    else:
        mask = ~xr.ufuncs.isnan(delta_h)

    # Terrain correction
    Ct = xr.where(
        mask,
        cte1 * (delta_h**2) / (distance**3) - cte2 * (delta_h**4) / (distance**5),
        0.0
    )

    return Ct.sum().item()


def plot_DataArray_percentiles(data_array, x='x', y='y', label='unit', title='DataArray_percentiles', cmap='magma', num_classes=10):
    """
    Plota um xarray.DataArray com cores graduadas em classes ajustáveis.

    Parâmetros:
        data_array (xr.DataArray): DataArray a ser plotado.
        x (str): Nome da coordenada para x (ou longitude).
        y (str): Nome da coordenada para y (ou latitude).
        label (str): Rótulo para a barra de cores.
        title (str): Título do gráfico.
        cmap (str): Colormap a ser utilizado.
        num_classes (int): Número de classes para dividir os dados (padrão: 10).
    """
    # Flatten the data to calculate percentiles, excluding NaN values
    values = data_array.values.flatten()

    # Calculando os percentis de acordo com o número de classes, excluindo NaNs
    percentiles = np.linspace(0, 100, num_classes + 1)
    classes = np.percentile(values[~np.isnan(values)], percentiles[1:-1])

    # Definindo cores para cada classe, mantendo os NaNs
    color_classes = np.digitize(data_array.values, classes, right=True)
    color_classes = np.where(np.isnan(data_array.values), np.nan, color_classes)

    # Criando o colormap discreto
    cmap_discrete = plt.get_cmap(cmap, num_classes)

    # Plotar o xarray.DataArray com o colormap discreto
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.pcolormesh(data_array[x], data_array[y], color_classes, cmap=cmap_discrete, shading='auto')

    # Criando a barra de cores com os rótulos dos valores das classes
    cbar = fig.colorbar(c, ax=ax, label=label, ticks=np.arange(1, num_classes + 1))
    class_labels = [f'< {classes[0]:.2f}'] + [f'{classes[i-1]:.2f} - {classes[i]:.2f}' for i in range(1, len(classes))]
    class_labels.append(f'> {classes[-1]:.2f}')
    cbar.ax.set_yticklabels(class_labels)

    # Configurações adicionais
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_aspect('equal', adjustable='box')

    # Ajustar o layout
    fig.tight_layout()
    plt.show()

def geotiff_to_dataarray(geotiff_path):
    """
    Convert a GeoTIFF file to an xarray.DataArray.

    Parameters:
        geotiff_path (str): Full path to the GeoTIFF file.

    Returns:
        xr.DataArray: DataArray with 'y' and 'x' coords.
    """
    # 1. check file exists
    if not os.path.exists(geotiff_path):
        raise FileNotFoundError(f"File not found: {geotiff_path}")

    # 2. open with rasterio
    with rasterio.open(geotiff_path) as src:
        array = src.read(1)           # first (or only) band
        transform = src.transform     # affine transform object
        crs = src.crs                 # coordinate reference system
        nodata = src.nodata

    # 3. compute coordinate vectors at pixel centers
    nrows, ncols = array.shape
    x0, dx = transform.c, transform.a  # origin x, pixel width
    y0, dy = transform.f, transform.e  # origin y, pixel height (usually negative)
    x = x0 + (np.arange(ncols) + 0.5) * dx
    y = y0 + (np.arange(nrows) + 0.5) * dy

    # 4. build the DataArray
    da = xr.DataArray(
        array,
        dims=('y', 'x'),
        coords={'x': x, 'y': y},
        attrs={'crs': crs.to_string(), 'transform': transform, 'nodata': nodata}
    )

    # 5. ensure y is ascending
    if da.y[0] > da.y[-1]:
        da = da.isel(y=slice(None, None, -1))

    return da


def associated_legendre_polynomial(n, m, t):
    """
    Calcula o polinômio de Legendre associado P_nm(t).

    Parâmetros:
        n (int): Grau do polinômio de Legendre.
        m (int): Ordem do polinômio de Legendre (0 <= m <= n).
        t (float): cos(theta).

    Retorno:
        float: Valor do polinômio de Legendre associado P_nm(t).
    """
    
    # Verifica se os tipos e valores de entrada são válidos
    if not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("n e m devem ser inteiros.")
    if m < 0 or m > n:
        raise ValueError("A ordem m deve satisfazer 0 <= m <= n.")
    
    # Índice máximo da soma com base na fórmula
    r = (n - m) // 2

    # Acumula os termos da soma
    total = 0.0
    for k in range(r + 1):
        numerator = (-1)**k * (t**(n - m - 2*k)) * math.factorial(2*n - 2*k)
        denominator = math.factorial(k) * math.factorial(n - k) * math.factorial(n - m - 2*k)
        total += numerator / denominator

    # Fator multiplicador fora da soma
    multiplier = (2**-n) * ((1 - t**2)**(m / 2))

    return multiplier * total


def legendre_polynomial(n, t):
    """
    Calcula o polinômio de Legendre P_n(t) usando a relação de recorrência:
    P_n(t) = -((n - 1)/n) * P_{n-2}(t) + ((2n - 1)/n) * t * P_{n-1}(t)
    
    Parâmetros:
        n (int): Grau do polinômio de Legendre.
        t (float ou np.array): Valor em que o polinômio será avaliado.
    
    Retorno:
        float ou np.array: Valor do polinômio de Legendre P_n(t) avaliado em t.
    """
    # Casos base: P_0(t) = 1 e P_1(t) = t
    if n == 0:
        return 1
    if n == 1:
        return t

    # Inicializa P_0 e P_1
    P_prev_prev = 1   # P_0
    P_prev = t        # P_1

    # Calcula P_k para k = 2, 3, ..., n usando a relação de recorrência
    for k in range(2, n + 1):
        P_current = -((k - 1) / k) * P_prev_prev + ((2 * k - 1) / k) * t * P_prev
        P_prev_prev, P_prev = P_prev, P_current

    return P_prev