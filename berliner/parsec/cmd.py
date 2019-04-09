from copy import deepcopy
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import numpy as np
from astropy.table import Table

str_cmd_welcome = """
Welcome to use berliner.parsec.cmd.CMD!
This module is to help you download CMD isochrones automatically.
Last modified: 2019.04.02

Homepage of CMD: http://stev.oapd.inaf.it/cgi-bin/cmd_3.2
Homepage of berliner: https://github.com/hypergravity/berliner
"""


class CMDParser(HTMLParser):
    """ CMD website parser """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starttags = []
        self.endtags = []
        self.data = []
        self.attrs = []
        self.default_kwargs = dict()

        # the options for photsys_file & imf_file
        self.photsys_file = []
        self.imf_file = []

        # the options for track_parsec & track_colibri
        self.track_parsec = []
        self.track_colibri = []

        # to grab the output file link
        self.output = None

    def handle_starttag(self, tag, attrs):
        self.starttags.append(tag)
        self.attrs.append(attrs)

        if len(attrs) >= 2:
            attr_dict = dict()
            for k, v in attrs:
                attr_dict[k] = v

            if "name" in attr_dict.keys() and "value" in attr_dict.keys():
                # ['checkbox', 'hidden', 'radio', 'submit', 'text']
                if attr_dict["type"] == "radio":
                    if "checked" in attr_dict.keys():
                        self.default_kwargs[attr_dict["name"]] = attr_dict["value"]

                    # extract parsec & colobri options
                    if attr_dict["name"] == "track_parsec":
                        self.track_parsec.append(attr_dict["value"])
                    if attr_dict["name"] == "track_colibri":
                        self.track_colibri.append(attr_dict["value"])

                elif attr_dict["type"] == "submit" or attr_dict["type"] == "hidden":
                    pass
                elif attr_dict["type"] == "checkbox":
                    self.default_kwargs[attr_dict["name"]] = attr_dict["value"]
                else:
                    self.default_kwargs[attr_dict["name"]] = attr_dict["value"]
                    #print(attr_dict)
            else:
                pass
                # print(attr_dict)
        elif len(attrs) == 1:
            # extract imf_file & photsys_file
            if attrs[0][0] == "value":
                if "tab_mag" in attrs[0][1]:
                    # photsys option
                    self.photsys_file.append(attrs[0][1].replace("tab_mag_odfnew/tab_mag_","").replace(".dat",""))
                if "tab_imf" in attrs[0][1]:
                    # imf option
                    self.imf_file.append(attrs[0][1].replace("tab_imf/imf_","").replace(".dat",""))

            # if used for output
            if attrs[0][0] == "href" and "../tmp/output" in attrs[0][1]:
                self.output = attrs[0][1].replace("..", "")

    def handle_endtag(self, tag):
        return

    def handle_data(self, data):
        return


def cmd_defaults(cmdhost, photsys_file="2mass_spitzer", imf_file="salpeter"):
    # get cmd web default keywords
    cmd_data = urlopen(cmdhost).read().decode('utf8')

    # parse cmd web
    cmdp = CMDParser()
    cmdp.feed(cmd_data)

    default_kwargs = cmdp.default_kwargs
    assert photsys_file in cmdp.photsys_file
    assert imf_file in cmdp.imf_file
    photsys_file = "tab_mag_odfnew/tab_mag_"+photsys_file+".dat"
    imf_file = "tab_imf/imf_"+imf_file+".dat"
    default_kwargs["photsys_file"] = photsys_file
    default_kwargs["imf_file"] = imf_file

    # complete for url
    default_kwargs["submit_form"] = "Submit"

    # adjust for unzipped file
    default_kwargs["output_gzip"] = "0"

    return default_kwargs, cmdp


class CMD:
    """ CMD class, to download isochrones automatically """
    def __init__(self):
        self.cmdhost = "http://stev.oapd.inaf.it"
        self.Zsun = 0.0152

        self.cmdp = None
        self.default_kwargs = None

        self.update()

        self.limit_mh = (-2.9,0.9)
        self.limit_logage = (1.1, 10.5)
        self.limit_z = (0.0000000000152, 0.1)

        self.photsys_file = self.cmdp.photsys_file
        self.imf_file = self.cmdp.imf_file
        self.track_parsec = self.cmdp.track_parsec
        self.track_colibri = self.cmdp.track_colibri

        # self.welcome()

    def help(self):
        print("-------------------------------------------------------")
        self.welcome()
        print("-------------------------------------------------------")
        print("Here are some hints for the options!")
        print("-------------------------------------------------------")
        print("track_parsec:")
        print(self.track_parsec)
        print("")
        print("track_colibri:")
        print(self.track_colibri)
        print("")
        print("imf_file:")
        print(self.imf_file)
        print("")
        print("photsys_file:")
        print(self.photsys_file)
        print("-------------------------------------------------------")

    def welcome(self):
        print(str_cmd_welcome)

    def update(self):
        """ update hosts """
        self.default_kwargs, self.cmdp = cmd_defaults(self.cmdhost + "/cgi-bin/cmd")

    def valid_logage(self, grid_logage):
        """ to validate grids of logAge, [M/H] """
        if grid_logage[0] < self.limit_logage[0] \
                or grid_logage[1] > self.limit_logage[1] \
                or grid_logage[2] < 0:
            raise Warning("@CMD: invalid logAge grid!")
            return False
        return True

    def valid_mh(self, grid_mh):
        if grid_mh[0] < self.limit_mh[0] \
                or grid_mh[1] > self.limit_mh[1] \
                or grid_mh[2] < 0:
            raise Warning("@CMD: invalid [M/H] grid!")
            return False
        return True

    def valid_z(self, grid_z):
        if grid_z[0] < self.limit_z[0] \
                or grid_z[1] > self.limit_z[1] \
                or grid_z[2] < 0:
            raise Warning("@CMD: invalid Z grid!")
            return False
        return True

    def get_one_isochrone(self, logage=9., z=0.0152, mh=None,
                          photsys_file="2mass_spitzer", imf_file="salpeter",
                          **kwargs):
        """ get one isochrone """
        # make url for request
        this_url = self.get_one_isochrone_url(logage=logage, z=z, mh=mh,
                                              photsys_file=photsys_file,
                                              imf_file=imf_file, **kwargs)

        # the output of request
        this_req = urlopen(this_url).read().decode('utf8')
        this_cmdp = CMDParser()
        this_cmdp.feed(this_req)
        this_output_link = this_cmdp.output
        if this_output_link is None:
            raise ValueError(
                "@CMD: no result for logage={} & z={} & [M/H]={}".format(logage,
                                                                         z, mh))
        this_output = urlopen(self.cmdhost + this_output_link).read().decode(
            'utf8').split("\n")

        # convert to table and return
        return convert_to_table(this_output)

    def get_isochrone_grid_mh(self, grid_logage=(8, 9, 0.1),
                              grid_mh=(-2.5, 0.5, 0.1),
                              photsys_file="2mass_spitzer",
                              imf_file="salpeter", n_jobs=1, verbose=10,
                              **kwargs):
        """ get isochrone grid

        Parameters
        ----------
        grid_logage:
            (logage_lo, logage_hi, logage_step)
        grid_mh:
            (mh_lo, mh_hi, mh_step)
        photsys_file:
            photometric system
        imf_file:
            IMF
        n_jobs:
            defaut is 1
        verbose:
            verbose for joblib
        kwargs:
            other keywords

        Returns
        -------
        meshflat_gird_logage:
            flat grid of logage
        meshflat_gird_mh:
            flat grid of [M/H]
        isocs:
            isochrone list
        """
        # validate grid
        if not self.valid_logage(grid_logage) or not self.valid_mh(grid_mh):
            return None, None, None

        # for isochrone grid, parallellized via logage
        n_logage = np.round((grid_logage[1] - grid_logage[0]) / grid_logage[2]) + 1
        n_mh = np.round((grid_mh[1] - grid_mh[0]) / grid_mh[2]) + 1
        print("@CMD: n(logAge)={:.0f}, n([M/H])={:.0f}".format(n_logage, n_mh))

        _grid_logage = np.arange(grid_logage[0], grid_logage[1]+grid_logage[2], grid_logage[2])
        _grid_mh = np.arange(grid_mh[0], grid_mh[1]+grid_mh[2], grid_mh[2])
        print("@CMD: grid(logAge): ", _grid_logage)
        print("@CMD: grid([M/H]) : ", _grid_mh)
        # grid_logage_for_parallel = (grid_logage[0],grid_logage[1],0)
        grid_mh_for_parallel = (grid_mh[0], grid_mh[1], grid_mh[2])
        #print(_grid_logage, _grid_mh)

        results = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(self.get_isochrone_set)(
                grid_logage=(_logage, _logage, 0), grid_mh=grid_mh_for_parallel,
                grid_z=None, photsys_file=photsys_file, imf_file=imf_file,
                **kwargs) for _logage in _grid_logage)
        isocs = []
        for _ in results:
            isocs.extend(_)

        meshflat_gird_logage = np.hstack(
            [_logage * np.ones_like(_grid_mh) for _logage in _grid_logage])
        meshflat_gird_mh = np.hstack([_grid_mh for _logage in _grid_logage])

        return meshflat_gird_logage, meshflat_gird_mh, isocs

    def get_isochrone_grid_true(self, tgrid_logage=(8.1, 8.2, 8.3),
                                tgrid_mh=None,
                                tgrid_z=None,
                                photsys_file="2mass_spitzer",
                                imf_file="salpeter", n_jobs=1, verbose=10,
                                **kwargs):
        """ get isochrone grid via true grid arrays

        Parameters
        ----------
        tgrid_logage:
            the true grid
        tgrid_mh:
            the true grid!
        tgrid_z:
            the true grid!
        photsys_file:
            photometric system
        imf_file:
            IMF
        n_jobs:
            defaut is 1
        verbose:
            verbose for joblib
        kwargs:
            other keywords

        Returns
        -------
        meshflat_gird_logage:
            flat grid of logage
        meshflat_gird_mh:
            flat grid of [M/H]
        isocs:
            isochrone list
        """

        if tgrid_z is not None:
            # [logage - Z] grid
            tgrid_logage = np.array(tgrid_logage)
            tgrid_z = np.array(tgrid_z)
            assert np.all(tgrid_logage > self.limit_logage[0])
            assert np.all(tgrid_logage < self.limit_logage[1])
            assert np.all(tgrid_z > self.limit_z[0])
            assert np.all(tgrid_z < self.limit_z[1])

            flat_logage, flat_z = [_.flatten() for _ in np.meshgrid(tgrid_logage, tgrid_z)]

            isocs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(self.get_one_isochrone)(
                    logage=_logage, z=_z, mh=None,
                    photsys_file=photsys_file, imf_file=imf_file,
                    **kwargs) for _logage, _z in zip(flat_logage, flat_z))

            return flat_logage, flat_z, isocs

        elif tgrid_mh is not None:
            # [logage - [M/H]] grid
            tgrid_logage = np.array(tgrid_logage)
            tgrid_mh = np.array(tgrid_mh)
            assert np.all(tgrid_logage > self.limit_logage[0])
            assert np.all(tgrid_logage < self.limit_logage[1])
            assert np.all(tgrid_mh > self.limit_mh[0])
            assert np.all(tgrid_mh < self.limit_mh[1])

            flat_logage, flat_mh = [_.flatten() for _ in np.meshgrid(tgrid_logage, tgrid_mh)]

            isocs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(self.get_one_isochrone)(
                    logage=_logage, z=None, mh=_mh,
                    photsys_file=photsys_file, imf_file=imf_file,
                    **kwargs) for _logage, _mh in zip(flat_logage, flat_mh))

            return flat_logage, flat_mh, isocs

        else:
            raise ValueError("@CMD: invalid input!")

    def get_isochrone_set(self, grid_logage=(8,9,0.1), grid_mh=(-2.5,0.5,0.1),
                          grid_z=None, photsys_file="2mass_spitzer", imf_file="salpeter",
                          **kwargs):
        """ to get a set of isochrones via web """
        # make url for request
        this_url = self.get_isochrone_set_url(grid_logage=grid_logage,
                                              grid_mh=grid_mh, grid_z=grid_z,
                                              photsys_file=photsys_file,
                                              imf_file=imf_file, **kwargs)

        # the output of request
        this_req = urlopen(this_url).read().decode('utf8')
        this_cmdp = CMDParser()
        this_cmdp.feed(this_req)
        this_output_link = this_cmdp.output
        if this_output_link is None:
            raise ValueError("@CMD: no result for logage={} & z={} & [M/H]={}".format(grid_logage, grid_z, grid_mh))
        this_output = urlopen(self.cmdhost + this_output_link).read().decode('utf8').split("\n")

        # convert to table and return
        return convert_to_table(this_output)

    def get_one_isochrone_url(self, logage=9., z=0.0152, mh=None,
                              photsys_file="2mass_spitzer", imf_file="salpeter",
                              **kwargs):
        """ generate url """
        # default keywords
        default_kwargs = deepcopy(self.default_kwargs)

        # photsys & imf
        assert photsys_file in self.photsys_file
        photsys_file = "tab_mag_odfnew/tab_mag_" + photsys_file + ".dat"
        imf_file = "tab_imf/imf_" + imf_file + ".dat"
        default_kwargs["photsys_file"] = photsys_file
        default_kwargs["imf_file"] = imf_file

        # age
        default_kwargs["isoc_isagelog"] = "1"
        default_kwargs["isoc_lagelow"] = str(logage)

        # metallicity
        if mh is not None:
            # use [M/H] for this request
            default_kwargs["isoc_ismetlog"] = "1"
            default_kwargs["isoc_metlow"] = str(mh)
        else:
            # use Z for this request
            default_kwargs["isoc_ismetlog"] = "0"
            default_kwargs["isoc_zlow"] = str(z)

        # other keywords
        default_kwargs.update(kwargs)

        # make url for request
        this_url = self.cmdhost + "/cgi-bin/cmd?" + urlencode(default_kwargs)

        return this_url

    def get_isochrone_set_url(self, grid_logage=(8,9,0.1), grid_mh=(-2.5,0.5,0.1),
                              grid_z=None, photsys_file="2mass_spitzer", imf_file="salpeter",
                              **kwargs):
        """ generate url """
        # default keywords
        default_kwargs = deepcopy(self.default_kwargs)

        # valid photsys_file
        assert photsys_file in self.photsys_file

        # photsys & imf
        photsys_file = "tab_mag_odfnew/tab_mag_" + photsys_file + ".dat"
        imf_file = "tab_imf/imf_" + imf_file + ".dat"
        default_kwargs["photsys_file"] = photsys_file
        default_kwargs["imf_file"] = imf_file

        # age
        default_kwargs["isoc_isagelog"] = "1"
        default_kwargs["isoc_lagelow"] = str(grid_logage[0])
        default_kwargs["isoc_lageupp"] = str(grid_logage[1])
        default_kwargs["isoc_dlage"] = str(grid_logage[2])

        # metallicity
        if grid_mh is not None:
            # use [M/H] for this request
            default_kwargs["isoc_ismetlog"] = "1"
            default_kwargs["isoc_metlow"] = str(grid_mh[0])
            default_kwargs["isoc_metupp"] = str(grid_mh[1])
            default_kwargs["isoc_dmet"] = str(grid_mh[2])
        else:
            # use Z for this request
            default_kwargs["isoc_ismetlog"] = "0"
            default_kwargs["isoc_zlow"] = str(grid_z[0])
            default_kwargs["isoc_zupp"] = str(grid_z[1])
            default_kwargs["isoc_dz"] = str(grid_z[2])

        # other keywords
        default_kwargs.update(kwargs)

        # make url for request
        this_url = self.cmdhost + "/cgi-bin/cmd?" + urlencode(default_kwargs)

        return this_url


def convert_to_table(this_output):
    """ convert isochrones to tables """
    line0 = np.where(["# Zini" in this_line for this_line in this_output])[0]
    #print(len(this_output), line0)
    line1 = np.int(np.where(["#isochrone terminated" in this_line for this_line in this_output])[0])
    line = np.append(line0, line1)
    n_isocs = len(line) - 1
    #print(n_isocs)

    if n_isocs == 1:
        # only one table
        isoc = Table.read(this_output[line[0]:line[1]], format="ascii.commented_header")
        return isoc
    elif n_isocs > 1:
        # multiple tables
        #print(line)
        isocs = [Table.read(this_output[line[i]:line[i+1]], format="ascii.commented_header") for i in range(len(line)-1)]
        return isocs
    else:
        raise ValueError(
            "@CMD: error when converting tables! no isochrone found, line0 = ", line0)
