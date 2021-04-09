from astrodendro import Dendrogram, ppp_catalog
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table
import pickle
import sys
import pandas as pd


"""
file_path = '/Users/shlomo/Desktop/Thesis/pythonProject/3D_Dust/perseus_3d_cutout.fits'
# image = fits.getdata(file_path)
wcs = WCS(fits.open(file_path)[0].header)
# d = Dendrogram.compute(image, min_value=25, min_delta=25, min_npix=150, verbose=True)
metadata = {'data_unit': u.cm ** (-3), 'spatial_scale': 1 * u.parsec, 'wcs': wcs}
# print("Finished dendrogramming, beginning table making")
d = Dendrogram.load_from('/Users/shlomo/Desktop/Thesis/pythonProject/Leike/Leike 25/Leike_Dend(25,25,150).fits')
# Produce tables
scheme_1 = ppp_catalog(d, metadata)
table_objects_deleted(scheme_1, d, min_mass=500)
scheme_1.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 25/Table(25)_filtered.fits')
table = ppp_catalog(d, metadata)
add_col_of_idx(table, scheme_1["_idx"])
table.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 25/Table(25).fits')

d = Dendrogram.load_from('/Users/shlomo/Desktop/Thesis/pythonProject/Leike/Leike 20/Leike_Dend(20,25,150).fits')
# Produce tables
scheme_1 = ppp_catalog(d, metadata)
table_objects_deleted(scheme_1, d, min_mass=500)
scheme_1.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 20/Table(20)_filtered.fits')
table = ppp_catalog(d, metadata)
add_col_of_idx(table, scheme_1["_idx"])
table.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 20/Table(20).fits')

d = Dendrogram.load_from('/Users/shlomo/Desktop/Thesis/pythonProject/Leike/Leike 15/Leike_Dend(15,25,150).fits')
# Produce tables
scheme_1 = ppp_catalog(d, metadata)
table_objects_deleted(scheme_1, d, min_mass=500)
scheme_1.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 15/Table(15)_filtered.fits')
table = ppp_catalog(d, metadata)
add_col_of_idx(table, scheme_1["_idx"])
table.write('/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike 25/Table(15).fits')
"""


class node:
    def __init__(self, idx, parent=None, children=None, descendents=None):
        self.idx = idx
        self.children = children
        self.parent = parent
        self.descendents = descendents


def row_number(table, idx):
    # returns equivalent/actual row number in the table b/c table row number changes.
    for i in range(len(table)):
        if table['_idx'][i] == idx:
            return i


def if_descendant_above_threshold(table, idx, structs, min_mass):
    # Orphan: object without a parent. Checks if the object (idx) has a descendant with > 1K SM, if so, return true.
    for element in range(len(structs)):
        if structs[element].idx == idx:
            for i in range(len(structs[element].descendants)):
                if table['mass'][structs[element].descendants[i].idx] >= min_mass:
                    return True
            break
    return False


def all_structs(dend):
    structs = []
    for i in range(len(dend.trunk)):
        structs.append(dend.trunk[i])
        for j in range(len(dend.trunk[i].descendants)):
            structs.append(dend.trunk[i].descendants[j])
    return structs


def list_idx(structs, idx):
    for index in range(len(structs)):
        if structs[index].idx == idx:
            return index


def table_objects_deleted(table, dend, min_mass):
    # Updates the table by removing objects with mass under min_mass and parents with children with that quality
    structs = all_structs(dend)
    rows_to_remove = []
    count = 0
    for row_num in range(len(table)):
        # first just remove those below min mass.
        if table['mass'][row_num] < min_mass:
            rows_to_remove.append(table['_idx'][row_num])
            structs.pop(list_idx(structs, row_num))
            count += 1
    # Now -- we want to keep leaves that: make sure they don't have a descendant above min_mass as well. if do,
    # remove them.
    print("length of all structs above min_mass in scheme 1 using length and count:", len(table) - len(structs))

    structs_idx = []
    for i in range(len(structs)):
        structs_idx.append(structs[i].idx)
    index = 0
    while index < len(structs):
        children = structs[index].descendants
        for j in range(len(children)):
            if children[j].idx in structs_idx:
                indexer = structs_idx.index(children[j].idx)
                structs.pop(indexer)
                structs_idx.pop(indexer)
                rows_to_remove.append(structs[indexer].idx)
                break
        index += 1

    for i in range(len(rows_to_remove)):
        table.remove_row(row_number(table, rows_to_remove[i]))


def table_objects_deleted_draft(table, dend, min_mass):
    # Updates the table by removing objects with mass under min_mass and parents with children with that quality
    structs = all_structs(dend)
    rows_to_remove = []
    for row in range(len(table)):
        # first just remove those below min mass.
        if table['mass'][row] < min_mass:
            rows_to_remove.append(table['_idx'][row])
        # Now -- we want to keep leaves that: make sure they don't have a descendant above min_mass as well. if do,
        # remove them.
        elif if_descendant_above_threshold(table, table['_idx'][row], structs, min_mass):
            rows_to_remove.append(table['_idx'][row])

    for i in range(len(rows_to_remove)):
        table.remove_row(row_number(table, rows_to_remove[i]))


def if_orphan(idx, nodes):
    # Orphan: object without a parent. Checks if the object (idx) is an orphan.
    i = 0
    while i < len(nodes):
        if nodes[i].idx == idx and nodes[i].parent is None:
            return True
        i += 1
    return False


def table_objects_deleted_2(nodes, table, min_mass=500):
    # Updates the table by removing objects with mass under min_mass and non-orphans.
    rows_to_remove = []
    for row in range(len(table)):
        if table['mass'][row] < min_mass:
            rows_to_remove.append(table['_idx'][row])
        elif not if_orphan(table['_idx'][row], nodes):
            rows_to_remove.append(table['_idx'][row])

    for i in range(len(rows_to_remove)):
        table.remove_row(row_number(table, rows_to_remove[i]))


def save_dend(d):
    # SAVE Dendrogram
    path_dend = '/Users/shlomo/Desktop/Thesis/pythonProject/Files created/pipeDend25,25,150.fits'
    d.save_to(path_dend)
    # Define Dendrogram
    file_path = '/Users/shlomo/Desktop/Thesis/pythonProject/Leike/leike_2020_xyz.fits'
    image = fits.getdata(file_path)
    wcs = WCS(fits.open(file_path)[0].header)
    d = Dendrogram.compute(image, min_value=25, min_delta=25, min_npix=150, verbose=True)
    metadata = {'data_unit': u.cm ** (-3), 'spatial_scale': 1 * u.parsec, 'wcs': wcs}
    print("Finished dendrogramming, beginning table making")


def compare_schemes(scheme_1, d, scheme_2):
    scheme_2 = Table.read('/Users/shlomo/Desktop/Thesis/pythonProject/LeikeTable(25,25,150).fits')
    print("1000 Counters: ")
    table_objects_deleted(scheme_1, d, min_mass=1000)
    table_objects_deleted_2(scheme_2, d, min_mass=1000)
    print("for min_mass 1000: ", len(scheme_1), len(scheme_2))
    print("500 Counters: ")
    scheme_1 = Table.read('/Users/shlomo/Desktop/Thesis/pythonProject/LeikeTable(25,25,150).fits')
    scheme_2 = Table.read('/Users/shlomo/Desktop/Thesis/pythonProject/LeikeTable(25,25,150).fits')
    table_objects_deleted(scheme_1, d, min_mass=500)
    table_objects_deleted_2(scheme_2, d, min_mass=500)
    print("for min_mass 500: ", len(scheme_1), len(scheme_2))
    scheme_1.write('')


def add_col_of_idx(table, idxes_exists):
    # fill it up with zeros for all, then change to ones where it exists.
    col = [0] * len(table)
    table.add_column(col, name='feature_exists', index=0)
    lst_of_idx = []
    for i in range(len(table)):
        lst_of_idx.append(table["_idx"][i])
    for i in range(len(idxes_exists)):
        if idxes_exists[i] in lst_of_idx:
            table["feature_exists"][lst_of_idx.index(idxes_exists[i])] = 1


def first_scheme(nodes, table, min_mass=500):
    rows_to_remove = []
    for row in range(len(table)):
        # first just remove those below min mass.
        if table['mass'][row] < min_mass:
            rows_to_remove.append(table['_idx'][row])
    index = 0
    while index < len(nodes):
        if nodes[index].idx in rows_to_remove:
            nodes.pop(index)
            index -= 1
        index += 1
    list_index = []
    for i in range(len(nodes)):
        list_index.append(nodes[i].idx)
    index = 0
    while index < len(nodes):
        for i in range(len(nodes[index].children)): #list not empty
            if nodes[index].children[i] in list_index:
                rows_to_remove.append(nodes[index].idx)
                nodes.pop(index)
                list_index.pop(index)
                index -= 1
                break
        index += 1
    # print(len(nodes))
    # print(len(rows_to_remove))
    for i in range(len(rows_to_remove)):
        table.remove_row(row_number(table, rows_to_remove[i]))
    # print(len(table))


def table_filtering():
    table = Table.read("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Leike_15/Table(15).fits")
    table.remove_column("feature_exists")
    table.write("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table_15.fits")
    file = open("/Users/shlomo/Desktop/Thesis/pythonProject/Code/nodes_dend_15.dat", 'rb')
    nodes = pickle.load(file)
    table_objects_deleted_2(nodes, table)
    # table is now updated per scheme 2.
    table.write("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table15(2Scheme).fits")
    table_original = Table.read("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table_15.fits")
    add_names(table_original, table["_idx"])
    table_original.write("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table15_withFeatures(2Scheme).fits")


    # New first scheme
    table = Table.read("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table_15.fits")
    first_scheme(nodes, table)
    table.write("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table15(1Scheme).fits")
    table_original = Table.read("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table_15.fits")
    add_names(table_original, table["_idx"])
    table_original.write("/Users/shlomo/Desktop/Thesis/pythonProject/Leike Updated Contents/Updated tables for 20,15/Table15_withFeatures(1Scheme).fits")


def check_features(nodes):
    list_index = []
    for i in range(len(nodes)):
        list_index.append(nodes[i].idx)

    num = 66
    print(nodes[list_index.index(num)].parent, nodes[list_index.index(num)].children, nodes[list_index.index(num)].descendents)


def to_significant_figures(num, n):
    # if isinstance(num, list):
    #     for i in range(len(num)):
    #         num[i] = '{:g}'.format(float('{:.{p}g}'.format(num[i], p=n)))
    #     return num
    # res = '{:g}'.format(float('{:.{p}g}'.format(num, p=n)))
    # return f'{res:g}'
    return ('%.2f' % num).rstrip('0').rstrip('.')

def to_significant_figures_int(num, n):
    if isinstance(num, list):
        for i in range(len(num)):
            num[i] = '{:g}'.format(float('{:.{p}g}'.format(num[i], p=n)))
        return num
    return '{:g}'.format(int('{:.{p}g}'.format(num, p=n)))

def scientific_figures(num):
    return "{:.2e}".format(num)



def find_apt_line(locations, catalog):
    # Returns line number and apt x,y,z in pc.
    # Locations is a 1d array with x,y, and z distances from solar system or a 1d array with l and b.
    line_numbers = []
    locationer = []
    difference = 20
    if len(locations) == 3:
        for i in range(len(catalog)):
            x, y, z = catalog["x_pc"][i], catalog["y_pc"][i], catalog["z_pc"][i]
            if x + difference > locations[0] > x - difference and y + difference > locations[1] > y - difference and z + difference > locations[2] > z - difference:
                line_numbers.append(catalog["_idx"][i])
                locationer.append([x, y, z])
        return line_numbers, locationer
    difference = 10
    if len(locations) == 2:
        for i in range(len(catalog)):
            l, b = catalog['l (deg)'][i], catalog['b (deg)'][i]
            if l + difference > locations[0] > l - difference and b + difference > locations[1] > b - difference:
                line_numbers.append(i)
                locationer.append([l, b])
        return line_numbers, locationer
    return "insufficient means"


def add_names(table, col_contents):
    # fill it up with zeros for all, then change to ones where it exists.
    col = []
    for i in range(len(table)):
        col.append("Cloud number {}".format(i))
    table.add_column(col, name='cloud', index=0)

    # create list of index for easier element detection
    lst_of_idx = []
    for i in range(len(table)):
        lst_of_idx.append(table["_idx"][i])

    for i in range(len(col_contents)):
        lst = col_contents[i]
        small_lst = lst[0]
        if not small_lst:
            continue
        for j in range(len(small_lst)):
            lst_of_idx.index(small_lst[j])
            table["cloud"][lst_of_idx.index(small_lst[j])] = lst[1]


def recognizing_clouds(cat):
    """
    This code is used to tag catalogs with names of clouds.
    """
    print("length in helpers", len(cat))
    # finding clouds in catalogs
    base_table = pd.read_excel(r'/Users/shlomo/Desktop/Thesis/pythonProject/Combined plots/Other '
                               r'catalogs/Table_Zucker.xlsx')

    names_col = []
    for cloud_number in range(len(base_table)):
        cloud_name = base_table["cloud"][cloud_number]
        # getting locations from catalog
        locations_xyz = [base_table["x_pc"][cloud_number], base_table["y_pc"][cloud_number],
                         base_table["z_pc"][cloud_number]]
        # locations_lb = [base_table['l'][cloud_number], base_table['b'][cloud_number]]
        result = find_apt_line(locations_xyz, cat)[0]
        names_col.append([result, cloud_name])
        print([result, cloud_name])

    add_names(cat, names_col)
    return cat
