# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2020 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from warnings import warn

import numpy as np
from nifty8 import MaskOperator

from nifty8.domain_tuple import DomainTuple
from nifty8.domains.power_space import PowerSpace
from nifty8.operators.adder import Adder
from nifty8.operators.contraction_operator import ContractionOperator
from nifty8.operators.distributors import PowerDistributor
from nifty8.operators.harmonic_operators import HarmonicTransformOperator
from nifty8.operators.normal_operators import LognormalTransform, NormalTransform
from nifty8.operators.simple_linear_operators import ducktape
from nifty8.operators.value_inserter import ValueInserter
from nifty8.sugar import full, makeField, makeOp
from nifty8.library.correlated_fields import (_log_vol, _Normalization,
                                _relative_log_k_lengths, _SlopeRemover,
                                _TwoLogIntegrations)

from nifty8.operators.diagonal_operator import DiagonalOperator

from nifty8.field import Field
from nifty8.operators.matrix_product_operator import MatrixProductOperator

from .custom_normal_operators import StandardUniformTransform

__all__ = ['CustomSimpleCorrelatedField']


def CustomSimpleCorrelatedField(
    target,
    offset_mean,
    offset_std,
    fluctuations,
    flexibility,
    asperity,
    loglogavgslope,
    use_uniform_prior_on_fluctuations,
    prefix="",
    harmonic_partner=None,
    op_to_apply_to_amp=None
):
    """Simplified version of :class:`~nifty8.library.correlated_fields.CorrelatedFieldMaker`.

    Assumes `total_N = 0`, `dofdex = None` and the presence of only one power
    spectrum, i.e. only one call of
    :func:`~nifty8.library.correlated_fields.CorrelatedFieldMaker.add_fluctuations`.

    See also
    --------
    * The simple correlated field model has first been described in "Comparison
      of classical and Bayesian imaging in radio interferometry", A&A 646, A84
      (2021) by P. Arras et al.
      `<https://doi.org/10.1051/0004-6361/202039258>`_

    Consider citing this paper, if you use the simple correlated field model.

    -- Parameters

    op_to_apply_to_amp: ift.OpChain
        An operator to apply to the amplitude spectrum before returning.

    """
    target = DomainTuple.make(target)
    if len(target) != 1:
        raise ValueError
    target = target[0]
    if harmonic_partner is None:
        harmonic_partner = target.get_default_codomain()
    else:
        target.check_codomain(harmonic_partner)
        harmonic_partner.check_codomain(target)
    for kk in (fluctuations, loglogavgslope):
        if len(kk) != 2:
            raise TypeError
    for kk in (offset_std, flexibility, asperity):
        if not (kk is None or len(kk) == 2):
            raise TypeError
    if flexibility is None and asperity is not None:
        raise ValueError
    if use_uniform_prior_on_fluctuations:
        fluct = StandardUniformTransform(key=prefix + 'fluctuations', N_copies=0, shift=0, upper_bound=1)
    else:
        fluct = LognormalTransform(*fluctuations, prefix + 'fluctuations', 0)
    avgsl = NormalTransform(*loglogavgslope, prefix + 'loglogavgslope', 0)

    pspace = PowerSpace(harmonic_partner)
    twolog = _TwoLogIntegrations(pspace)
    expander = ContractionOperator(twolog.domain, 0).adjoint
    ps_expander = ContractionOperator(pspace, 0).adjoint
    vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))
    slope = vslope @ ps_expander @ avgsl
    a = slope

    if flexibility is not None:
        flex = LognormalTransform(*flexibility, prefix + 'flexibility', 0)
        dom = twolog.domain[0]
        vflex = np.empty(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(pspace))
        vflex = makeOp(makeField(dom, vflex))
        sig_flex = vflex @ expander @ flex
        xi = ducktape(dom, None, prefix + 'spectrum')

        shift = np.empty(dom.shape)
        shift[0] = _log_vol(pspace)**2 / 12.
        shift[1] = 1
        shift = makeField(dom, shift)

        if asperity is None:
            asp = makeOp(shift.ptw("sqrt")) @ (xi*sig_flex)
        else:
            asp = LognormalTransform(*asperity, prefix + 'asperity', 0)
            vasp = np.empty(dom.shape)
            vasp[0] = 1
            vasp[1] = 0
            vasp = makeOp(makeField(dom, vasp))
            sig_asp = vasp @ expander @ asp
            asp = xi*sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")

        a = a + _SlopeRemover(pspace, 0) @ twolog @ asp
    a = _Normalization(pspace, 0) @ a
    maskzm = np.ones(pspace.shape)
    maskzm[0] = 0
    maskzm = makeOp(makeField(pspace, maskzm))
    a = (maskzm @ ( ( ps_expander @ fluct ) * a ) )
    if offset_std is not None:
        zm = LognormalTransform(*offset_std, prefix + 'zeromode', 0)
        insert = ValueInserter(pspace, (0,))
        a = a + insert(zm)
    a = a.scale(target.total_volume)

    ht = HarmonicTransformOperator(harmonic_partner, target)
    pd = PowerDistributor(harmonic_partner, pspace)
    xi = ducktape(harmonic_partner, None, prefix + 'xi')

    if op_to_apply_to_amp is not None:
        amp_op = op_to_apply_to_amp[0]
        mode = op_to_apply_to_amp[1]
        apply_xi_s_envelope = op_to_apply_to_amp[2]

        amplitude_spectrum_xi_s_envelope = np.loadtxt("pipe_1_xi_s_amp_spec_envelope_y.txt")
        amplitude_spectrum_xi_s_envelope_freqs = np.loadtxt("pipe_1_xi_s_amp_spec_envelope_x.txt")

        from scipy.interpolate import interp1d
        amplitude_spectrum_xi_s_envelope_callable = interp1d(x=amplitude_spectrum_xi_s_envelope_freqs,
                                                             y=amplitude_spectrum_xi_s_envelope, kind="linear",
                                                             assume_sorted=False, fill_value="extrapolate")

        x_field = pspace.k_lengths
        special_amp_xi_s_envelope = amplitude_spectrum_xi_s_envelope_callable(x_field)
        special_amp_xi_s_envelope_field = makeField(pspace, arr=special_amp_xi_s_envelope)
        special_amp_xi_s_envelope_diag_op = DiagonalOperator(special_amp_xi_s_envelope_field)

        if mode == "add":
            a = a + amp_op
        elif mode == "multiply":
            a = a * amp_op
        else:
            raise ValueError("Unknown mode '%s'" % mode, " needs to be either 'add' or 'multiply'")

        if apply_xi_s_envelope:
            print("\nAdding fixed xi_s envelope to amplitude operator inside simple correlated field code \n")
            a = special_amp_xi_s_envelope_diag_op(a)
        else:
            pass
    else:
        pass

    op = ht(pd(a).real*xi)

    if offset_mean is not None:
        op = Adder(full(op.target, float(offset_mean))) @ op
    op.amplitude = a
    op.power_spectrum = a**2
    return op

def VerySimpleCorrelatedField(
        target,
        fluctuations,
        loglogavgslope,
        prefix="",
        harmonic_partner=None,
        override_with_exact_values=False,
        custom_harmonic_xi_s=None,
):
    """Simplified version of :class:`~nifty8.library.correlated_fields.CorrelatedFieldMaker`.

    Assumes `total_N = 0`, `dofdex = None` and the presence of only one power
    spectrum, i.e. only one call of
    :func:`~nifty8.library.correlated_fields.CorrelatedFieldMaker.add_fluctuations`.

    See also
    --------
    * The simple correlated field model has first been described in "Comparison
      of classical and Bayesian imaging in radio interferometry", A&A 646, A84
      (2021) by P. Arras et al.
      `<https://doi.org/10.1051/0004-6361/202039258>`_

    Consider citing this paper, if you use the simple correlated field model.
    """
    target = DomainTuple.make(target)
    if len(target) != 1:
        raise ValueError
    target = target[0]
    if harmonic_partner is None:
        harmonic_partner = target.get_default_codomain()
    else:
        target.check_codomain(harmonic_partner)
        harmonic_partner.check_codomain(target)

    if not override_with_exact_values:
        for kk in (fluctuations, loglogavgslope):
            if len(kk) != 2:
                raise TypeError

    if override_with_exact_values:
        fluct = fluctuations
        avgsl = loglogavgslope
    else:
        # fluct = StandardUniformTransform(key=prefix + 'fluctuations', N_copies=0, shift=0, upper_bound=0.5)
        fluct = LognormalTransform(*fluctuations, prefix + 'fluctuations', 0)
        avgsl = NormalTransform(*loglogavgslope, prefix + 'loglogavgslope', 0)

    pspace = PowerSpace(harmonic_partner)
    ps_expander = ContractionOperator(pspace, 0).adjoint
    vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))
    if override_with_exact_values:
        fld = Field(DomainTuple.make(ps_expander.domain), val=np.array(avgsl))
        step1 = ps_expander(fld)
        slope = vslope(step1)  # field
    else:
        slope = vslope @ ps_expander @ avgsl
    a = slope

    if override_with_exact_values:
        a = _Normalization(pspace, 0)(a)
    else:
        a = _Normalization(pspace, 0) @ a
    maskzm = np.ones(pspace.shape)
    maskzm[0] = 0
    maskzm = makeOp(makeField(pspace, maskzm))
    if override_with_exact_values:
        fld = Field(DomainTuple.make(ps_expander.domain), val=np.array(fluct))
        a = (maskzm(((ps_expander(fld)) * a)))
    else:
        a = (maskzm @ ((ps_expander @ fluct) * a))
    a = a.scale(target.total_volume)

    ht = HarmonicTransformOperator(harmonic_partner, target)
    pd = PowerDistributor(harmonic_partner, pspace)
    xi = ducktape(harmonic_partner, None, prefix + 'xi')

    ## ---- FINAL OPERATOR ---- ##
    # op = ht(side_step.real * xi)
    lp = target.shape[0]
    if custom_harmonic_xi_s is not None:
        harmonic_xi_s = custom_harmonic_xi_s.copy()
    else:
        harmonic_xi_s = np.random.normal(0, 1, lp) + 1j * np.random.normal(0, 1, lp)
        # harmonic_xi_s = np.random.normal(0, 1, lp)


    xi_fld = Field(DomainTuple.make(pd(a).domain), val=harmonic_xi_s)

    op = ht(pd(a)*xi_fld)

    ## ---- ATTRIBUTES ---- ##
    op._domain = DomainTuple.make(target)
    op.amplitude = a
    op.power_spectrum = a ** 2

    return op

def SimpleNonStationaryField(
    target,
    offset_mean,
    offset_std,
    fluctuations,
    flexibility,
    asperity,
    loglogavgslope,
    prefix="",
    harmonic_partner=None,
    second_power_spectrum=False,
    num_of_pixels_to_truncate = None,
    full_matrix = False
):
    """Simplified version of :class:`~nifty8.library.correlated_fields.CorrelatedFieldMaker`.

    Assumes `total_N = 0`, `dofdex = None` and the presence of only one power
    spectrum, i.e. only one call of
    :func:`~nifty8.library.correlated_fields.CorrelatedFieldMaker.add_fluctuations`.

    See also
    --------
    * The simple correlated field model has first been described in "Comparison
      of classical and Bayesian imaging in radio interferometry", A&A 646, A84
      (2021) by P. Arras et al.
      `<https://doi.org/10.1051/0004-6361/202039258>`_

    Consider citing this paper, if you use the simple correlated field model.
    """
    target = DomainTuple.make(target)
    if len(target) != 1:
        raise ValueError
    target = target[0]
    if harmonic_partner is None:
        harmonic_partner = target.get_default_codomain()
    else:
        target.check_codomain(harmonic_partner)
        harmonic_partner.check_codomain(target)
    for kk in (fluctuations, loglogavgslope):
        if len(kk) != 2:
            raise TypeError
    for kk in (offset_std, flexibility, asperity):
        if not (kk is None or len(kk) == 2):
            raise TypeError
    if flexibility is None and asperity is not None:
        raise ValueError
    fluct = LognormalTransform(*fluctuations, prefix + 'fluctuations', 0)
    avgsl = NormalTransform(*loglogavgslope, prefix + 'loglogavgslope', 0)

    pspace = PowerSpace(harmonic_partner)
    twolog = _TwoLogIntegrations(pspace)
    expander = ContractionOperator(twolog.domain, 0).adjoint
    ps_expander = ContractionOperator(pspace, 0).adjoint
    vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))
    slope = vslope @ ps_expander @ avgsl
    a = slope

    if flexibility is not None:
        flex = LognormalTransform(*flexibility, prefix + 'flexibility', 0)
        dom = twolog.domain[0]
        vflex = np.empty(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(pspace))
        vflex = makeOp(makeField(dom, vflex))
        sig_flex = vflex @ expander @ flex
        xi = ducktape(dom, None, prefix + 'spectrum')

        shift = np.empty(dom.shape)
        shift[0] = _log_vol(pspace)**2 / 12.
        shift[1] = 1
        shift = makeField(dom, shift)
        if asperity is None:
            asp = makeOp(shift.ptw("sqrt")) @ (xi*sig_flex)
        else:
            asp = LognormalTransform(*asperity, prefix + 'asperity', 0)
            vasp = np.empty(dom.shape)
            vasp[0] = 1
            vasp[1] = 0
            vasp = makeOp(makeField(dom, vasp))
            sig_asp = vasp @ expander @ asp
            asp = xi*sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")
        a = a + _SlopeRemover(pspace, 0) @ twolog @ asp
    a = _Normalization(pspace, 0) @ a
    maskzm = np.ones(pspace.shape)
    maskzm[0] = 0
    maskzm = makeOp(makeField(pspace, maskzm))
    a = (maskzm @ ((ps_expander @ fluct)*a))
    if offset_std is not None:
        zm = LognormalTransform(*offset_std, prefix + 'zeromode', 0)
        insert = ValueInserter(pspace, (0,))
        a = a + insert(zm)
    a = a.scale(target.total_volume)

    print("Constructing harmonic operator...")
    ht = HarmonicTransformOperator(harmonic_partner, target)
    print("Constructed harmonic operator")
    pd = PowerDistributor(harmonic_partner, pspace)
    xi = ducktape(harmonic_partner, None, prefix + 'xi')
    op = ht(pd(a).real*xi)
    if offset_mean is not None:
        op = Adder(full(op.target, float(offset_mean))) @ op
    op.amplitude = a
    op.power_spectrum = a**2

    if full_matrix is True:
        n = target.shape[0]
        # mat = 1/n**2*np.arange(n**2).reshape(n, n) + 0.1*np.random.rand(n**2).reshape((n,n,))
        mat = np.diag(np.array([1/(i**(1/3)+1) for i in range(n)]))
        matrix = MatrixProductOperator(target, matrix=mat)
        return matrix
    elif second_power_spectrum is True:

        fluctuations = (fluctuations[0]*2, fluctuations[1]*2)
        loglogavgslope = (loglogavgslope[0]-1, loglogavgslope[1])

        fluct2 = LognormalTransform(*fluctuations, prefix + 'fluctuations', 0)
        avgsl2 = NormalTransform(*loglogavgslope, prefix + 'loglogavgslope', 0)

        slope2 = vslope @ ps_expander @ avgsl2

        a2 = slope2

        if flexibility is not None:
            flex2 = LognormalTransform(*flexibility, prefix + 'flexibility2', 0)
            dom2 = twolog.domain[0]
            vflex2 = np.empty(dom.shape)
            vflex2[0] = vflex2[1] = np.sqrt(_log_vol(pspace))
            vflex2 = makeOp(makeField(dom, vflex))
            sig_flex2 = vflex @ expander @ flex
            xi2 = ducktape(dom, None, prefix + 'spectrum2')

            shift2 = np.empty(dom.shape)
            shift2[0] = _log_vol(pspace) ** 2 / 12.
            shift2[1] = 1
            shift2 = makeField(dom, shift)
            if asperity is None:
                asp2 = makeOp(shift.ptw("sqrt")) @ (xi2 * sig_flex2)
            else:
                asp2 = LognormalTransform(*asperity, prefix + 'asperity2', 0)
                vasp2 = np.empty(dom.shape)
                vasp2[0] = 1
                vasp2[1] = 0
                vasp2 = makeOp(makeField(dom, vasp2))
                sig_asp2 = vasp2 @ expander @ asp2
                asp2 = xi2 * sig_flex2 * (Adder(shift) @ sig_asp2).ptw("sqrt")
            a2 = a2 + _SlopeRemover(pspace, 0) @ twolog @ asp2
        a2 = _Normalization(pspace, 0) @ a2
        maskzm2 = np.ones(pspace.shape)
        maskzm2[0] = 0
        maskzm2 = makeOp(makeField(pspace, maskzm2))
        a2 = (maskzm2 @ ((ps_expander @ fluct2) * a2))
        if offset_std is not None:
            zm2 = LognormalTransform(*offset_std, prefix + 'zeromode', 0)
            insert2 = ValueInserter(pspace, (0,))
            a2 = a2 + insert2(zm2)
        a2 = a2.scale(target.total_volume)


        op2 = ht(pd(a2).real * xi)
        flags = [True]*target.shape[0]

        flags = np.array(flags)
        num_of_pixels_to_truncate = np.array(num_of_pixels_to_truncate)

        flags[-num_of_pixels_to_truncate:] = False
        mask = CustomMaskOperator(flags=Field(DomainTuple.make(target), flags), target=target)

        return op + mask @ op2

    return op


# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from nifty8.domain_tuple import DomainTuple
from nifty8.domains.unstructured_domain import UnstructuredDomain
from nifty8.field import Field
from nifty8.operators.linear_operator import LinearOperator


class CustomMaskOperator(LinearOperator):
    """Implementation of a mask response

    Takes a field, applies flags and returns the values of the field in a
    :class:`UnstructuredDomain`.

    Parameters
    ----------
    flags : :class:`nifty8.field.Field`
        Is converted to boolean. Where True, the input field is flagged.
    """
    def __init__(self, flags, target):
        if not isinstance(flags, Field):
            raise TypeError
        self._domain = DomainTuple.make(flags.domain)
        self._flags = flags.val
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        # if mode == self.TIMES:
        res = np.where(self._flags, x, 0)
        return Field(self.target, res)
        # res = np.empty(self.domain.shape, x.dtype)
        # res[self._flags] = x
        # res[~self._flags] = 0
        # return Field(self.domain, res)


class CustomMaskOperator2(LinearOperator):
    """Implementation of a mask response

    Takes a field and returns the same field in the same domain type but the n values at the beginning cut out

    Parameters
    ----------
    flags : :class:`nifty8.field.Field`
        Is converted to boolean. Where True, the input field is flagged.
    """
    def __init__(self, target, how_many_first_pixels_to_cut):
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._n = how_many_first_pixels_to_cut

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        # if mode == self.TIMES:
        dom = self._target
        old_shape = self._target.shape[0]
        to_cut = np.sum(array == False)  # how many pixels at the beginning to cut
        new_domain = DomainTuple.make(RGSpace(shape=(old_shape-to_cut, ), distances=dom[0].distances))
        res = np.where(self._flags, x, 0)
        return Field(new_domain, x[self._n:])
        # res = np.empty(self.domain.shape, x.dtype)
        # res[self._flags] = x
        # res[~self._flags] = 0
        # return Field(self.domain, res)

def customCFM(target,
    offset_mean,
    offset_std,
    fluctuations,
    flexibility,
    asperity,
    loglogavgslope,
    prefix="",
    harmonic_partner=None):
    target = DomainTuple.make(target)
    if len(target) != 1:
        raise ValueError
    target = target[0]
    if harmonic_partner is None:
        harmonic_partner = target.get_default_codomain()
    else:
        target.check_codomain(harmonic_partner)
        harmonic_partner.check_codomain(target)
    for kk in (fluctuations, loglogavgslope):
        if len(kk) != 2:
            raise TypeError
    for kk in (offset_std, flexibility, asperity):
        if not (kk is None or len(kk) == 2):
            raise TypeError
    if flexibility is None and asperity is not None:
        raise ValueError
    fluct = StandardUniformTransform(key=prefix + 'fluctuations', N_copies=0, upper_bound=fluctuations[0], shift=fluctuations[1])
    avgsl = StandardUniformTransform(key=prefix + 'loglogavgslope', N_copies=0, upper_bound=loglogavgslope[0], shift=loglogavgslope[1])

    pspace = PowerSpace(harmonic_partner)
    twolog = _TwoLogIntegrations(pspace)
    expander = ContractionOperator(twolog.domain, 0).adjoint
    ps_expander = ContractionOperator(pspace, 0).adjoint
    vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))
    slope = vslope @ ps_expander @ avgsl
    a = slope

    if flexibility is not None:
        flex = LognormalTransform(*flexibility, prefix + 'flexibility', 0)
        dom = twolog.domain[0]
        vflex = np.empty(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(pspace))
        vflex = makeOp(makeField(dom, vflex))
        sig_flex = vflex @ expander @ flex
        xi = ducktape(dom, None, prefix + 'spectrum')

        shift = np.empty(dom.shape)
        shift[0] = _log_vol(pspace) ** 2 / 12.
        shift[1] = 1
        shift = makeField(dom, shift)
        if asperity is None:
            asp = makeOp(shift.ptw("sqrt")) @ (xi * sig_flex)
        else:
            asp = LognormalTransform(*asperity, prefix + 'asperity', 0)
            vasp = np.empty(dom.shape)
            vasp[0] = 1
            vasp[1] = 0
            vasp = makeOp(makeField(dom, vasp))
            sig_asp = vasp @ expander @ asp
            asp = xi * sig_flex * (Adder(shift) @ sig_asp).ptw("sqrt")
        a = a + _SlopeRemover(pspace, 0) @ twolog @ asp
    a = _Normalization(pspace, 0) @ a
    maskzm = np.ones(pspace.shape)
    maskzm[0] = 0
    maskzm = makeOp(makeField(pspace, maskzm))
    a = (maskzm @ ((ps_expander @ fluct) * a))
    if offset_std is not None:
        zm = LognormalTransform(*offset_std, prefix + 'zeromode', 0)
        insert = ValueInserter(pspace, (0,))
        a = a + insert(zm)
    a = a.scale(target.total_volume)

    ht = HarmonicTransformOperator(harmonic_partner, target)
    pd = PowerDistributor(harmonic_partner, pspace)
    xi = ducktape(harmonic_partner, None, prefix + 'xi')
    op = ht(pd(a).real * xi)
    if offset_mean is not None:
        op = Adder(full(op.target, float(offset_mean))) @ op
    op.amplitude = a
    op.power_spectrum = a ** 2

    return op

from nifty8.operators.field_zero_padder import FieldZeroPadder

def weirdCFM(target,
    offset_mean,
    offset_std,
    fluctuations,
    flexibility,
    asperity,
    loglogavgslope,
    prefix="",
    harmonic_partner=None):
    n = target.shape[0]
    ops = []
    # for i in range(n) - 100:
    for i in range(3):
        print("COMPUTING ", i ,"th OPERATOR")
        # dont compute elements on the last 100th off-diagonals
        new_target = RGSpace_cutter(target, i)
        diag_operator = customCFM(
            new_target,
            offset_mean,
            offset_std,
            fluctuations,
            flexibility,
            asperity,
            loglogavgslope,
            prefix="",
            harmonic_partner=None
        )

        # first mask incoming xi_s, then apply diagonal operator, then zero_padded xi_s at the end
        mask = CustomMaskOperator2(target=target, how_many_first_pixels_to_cut=i)
        X = FieldZeroPadder(domain=new_target, new_shape=(n,))
        intermediate_op = X @ diag_operator @ mask
        ops.append(intermediate_op)

    return sum(ops)

from nifty8.domains import RGSpace

def RGSpace_cutter(rg_space, to_cut):
    n = rg_space.shape[0]
    distances = rg_space[0].distances
    print("n , distances : ", n, distances)
    return RGSpace((n-to_cut,), distances)