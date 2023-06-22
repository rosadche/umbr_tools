# PLUMED EXTERNAL Input File Generation
# Reilly Osadchey Brown

import pandas as pd
import numpy as np
import scipy 

def plumed_ext_from_fes_1d(file, outfile, cv_name, periodic, bias_name, pad_num=25, s=0):
    """ Uses a Scipy Cubic Spline to easily write the PLUMED external file """
    
    if isinstance(periodic, bool):
        if periodic:
            periodic_str = "true"
        else:
            periodic_str = "false"
    else:
        raise Exception("Periodic must be a Boolean (True or False)!")
    
    #===============================================================================
    #                     READ FES FILE & GET DERIV
    #===============================================================================
    
    df = pd.read_csv(file, sep='\s+', comment="#")
    cv = df["bin"].to_numpy()
    fes_orig = df["f"].to_numpy()
    min_cv = cv.min()
    max_cv = cv.max()
    
    ext = 3 # return last and first values for values outside original range
    orig_spline     = scipy.interpolate.UnivariateSpline(cv, fes_orig, k=3, s=s, ext=ext)
    
    cv_delta = cv[1] - cv[0]
    npoints = len(cv) + pad_num * 2
    nbins = npoints - 1
    new_xs = np.linspace(min_cv-pad_num*cv_delta, max_cv+pad_num*cv_delta, npoints, endpoint=True)
    
    fes_spline = orig_spline(new_xs)
    fes_spline -= fes_spline.min()
    df_dx_spline = orig_spline.derivative(n=1)(new_xs)
    
    cv_min = new_xs.min()
    cv_max = new_xs.max()
    
    #===============================================================================
    #                           WRITE PLUMED GRID
    #===============================================================================
    header = ""
    header += "#! " + f"FIELDS {cv_name} {bias_name}.bias der_{cv_name}" + "\n"
    header += "#! " + f"SET min_{cv_name} {cv_min}" + "\n"
    header += "#! " + f"SET max_{cv_name} {cv_max}" + "\n"
    header += "#! " + f"SET nbins_{cv_name} {nbins}" + "\n"
    header += "#! " + f"SET periodic_{cv_name} {periodic_str}" + "\n"
    
    for i in range(npoints):
        cv_val = new_xs[i]
        f      = fes_spline[i]
        df_dcv = df_dx_spline[i]
        header += f"{cv_val:.8f} {f:.8f} {df_dcv:.8f}" + "\n"
    
    with open(outfile, "w") as f:
        f.write(header)



def plumed_ext_from_fes_2d(file, outfile, cv_name, periodic, bias_name, pad_num=25):
    """ """
    
    # Check to make sure cv_name, periodic are tuples/lists of length 2
    checks = [cv_name, periodic]
    for var in checks:
        if not isinstance(var, (list, tuple)):
            raise Exception(f"{var} must be list/tuple!")
        else:
            if len(var) != 2:
                raise Exception(f"{var} must be of length 2 for the 2 dimensions!")
    
    cv_name_x = cv_name[0]
    cv_name_y = cv_name[1]
    
    per_str_lst = []
    for per_val in periodic:
        if isinstance(per_val, bool):
            if per_val:
                per_str_lst.append("true")
            else:
                per_str_lst.append("false")
        else:
            raise Exception("Periodic must contain Booleans (True or False)!")
                
    #===============================================================================
    #                     READ FES FILE & GET DERIV
    #===============================================================================
    df = pd.read_csv(file, sep='\s+', comment="#")
    df_pivot = df.pivot_table(index=cv_name[0], columns=cv_name[1], values="f")
    Z = df_pivot.T.values
    X = df_pivot.index.values
    Y = df_pivot.columns.values
    
    unique_x = np.unique(X)
    unique_y = np.unique(Y)
    
    min_cv_x = unique_x.min()
    max_cv_x = unique_x.max()
    min_cv_y = unique_y.min()
    max_cv_y = unique_y.max()
    
    cv_delta_x = unique_x[1] - unique_x[0]
    cv_delta_y = unique_y[1] - unique_y[0]
    
    # padding
    # need to pad the grid so plumed doesn't crash if the simualtion tries to leave the grid
    new_x = np.linspace(min_cv_x - pad_num*cv_delta_x ,  max_cv_x + pad_num*cv_delta_x, num=(2*pad_num)+len(unique_x), endpoint=True)
    new_y = np.linspace(min_cv_y - pad_num*cv_delta_y ,  max_cv_y + pad_num*cv_delta_y, num=(2*pad_num)+len(unique_y), endpoint=True)
    new_grid = np.meshgrid(new_x, new_y)
    
    try:
        # In order to create a regular padded grid, we will loop over all current rows which are
        # at least partially occupied then use a 1d spline to pad and interpolate missing values
        # then we will do the same for the resulting output but loop over columns
        Z_new = np.empty( (len(unique_y), len(new_x)) )
        for row in range(Z.shape[0]):
            #loop extrapolate that row constants right and left
            # cubic spline, constant extrapolation, no smoothing
            # Z includes nan values, need to drop those on a row by row basis or they will mess with the spline
            Z_no_nan    = Z[row, :][~np.isnan(Z[row, :])]
            X_no_nan    = X[~np.isnan(Z[row, :])]
            row_spline  = scipy.interpolate.UnivariateSpline(X_no_nan, Z_no_nan, k=3, s=0, ext=3)
            row_vals    = row_spline(new_x)
            Z_new[row, :] = row_vals
        
        Z_new_new = np.empty_like(new_grid[0])
        for col in range(Z_new.shape[1]):
            # loop extrapolate that column constants up and down
            # cubic spline, constant extrapolation, no smoothing
            # Z includes nan values, need to drop those on a row by row basis or they will mess with the spline
            Z_no_nan    = Z_new[:, col][~np.isnan(Z_new[:, col])]
            Y_no_nan    = Y[~np.isnan(Z_new[:, col])]
            col_spline  = scipy.interpolate.UnivariateSpline(Y_no_nan, Z_no_nan, k=3, s=0, ext=3)
            col_vals    = col_spline(new_y)
            Z_new_new[:, col] = col_vals
    except:
        print("Scipy 1d interpolation failed, trying nearest neightbor 2d interpolation...")
        
        Z_new_new = np.full_like(new_grid[0], np.nan)
        Z_new_new[pad_num:Z.shape[0]+pad_num, pad_num:Z.shape[1]+pad_num] = Z
         
        # fill any missing off the new, full-size grid
        Z_new_new = manual_2d_nearest_interpolation(Z_new_new)
        
    #The first result in returned tuple is the gradient calculated with respect to the columns (down a column) 
    # and the one that follows is the gradient calculated with respect to the rows (across a row)
    df_dy, df_dx = np.gradient(Z_new_new, new_x, new_y)
    
    # Finish up getting all required variables
    npoints_x = len(new_x)
    nbins_x = npoints_x - 1
    npoints_y = len(new_y)
    nbins_y = npoints_y - 1
    
    #===============================================================================
    #                           WRITE PLUMED GRID
    #===============================================================================
    header = ""
    header += "#! " + f"FIELDS {cv_name_x} {cv_name_y} {bias_name}.bias der_{cv_name_x} der_{cv_name_y}" + "\n"
    header += "#! " + f"SET min_{cv_name_x} {new_x.min()}" + "\n"
    header += "#! " + f"SET max_{cv_name_x} {new_x.max()}" + "\n"
    header += "#! " + f"SET nbins_{cv_name_x} {nbins_x}" + "\n"
    header += "#! " + f"SET periodic_{cv_name_x} {per_str_lst[0]}" + "\n"
    header += "#! " + f"SET min_{cv_name_y} {new_y.min()}" + "\n"
    header += "#! " + f"SET max_{cv_name_y} {new_y.max()}" + "\n"
    header += "#! " + f"SET nbins_{cv_name_y} {nbins_y}" + "\n"
    header += "#! " + f"SET periodic_{cv_name_y} {per_str_lst[1]}" + "\n"
    
    # Order default is C, but this means we get changing x_var, constant y_var in output
    # which is incompatible with plumed.
    # order=F makes it constant x_var, changing y_var which we need
    for i in range(len(Z_new_new.ravel(order="F"))):
        x_val   = new_grid[0].ravel(order="F")[i]
        y_val   = new_grid[1].ravel(order="F")[i]
        f_val   = Z_new_new.ravel(order="F")[i]
        df_dx_val = df_dx.ravel(order="F")[i]
        df_dy_val = df_dy.ravel(order="F")[i]
        header += f"{x_val:.8f} {y_val:.8f} {f_val:.8f} {df_dx_val:.8f} {df_dy_val:.8f}" + "\n"
    
    with open(outfile, "w") as f:
        f.write(header)


def manual_2d_nearest_interpolation(Z):
    from scipy.spatial import KDTree
    
    a = np.ma.masked_invalid(Z, copy=True) # mask where nan occur
    # data must b nxm n=points, m=dimension of space

    x,y=np.mgrid[0:a.shape[0],0:a.shape[1]]
    
    xygood = np.array((x[~a.mask],y[~a.mask])).T
    xybad = np.array((x[a.mask],y[a.mask])).T
    
    a[a.mask] = a[~a.mask][KDTree(xygood).query(xybad)[1]]
    return a
            
    
    
    
    