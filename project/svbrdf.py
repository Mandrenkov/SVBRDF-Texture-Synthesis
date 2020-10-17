from __future__ import annotations  # type: ignore

import abc
import math
import torch
import vector
import utils

from torch import Tensor
from typing import Callable, List, Tuple


class SVBRDF(abc.ABC):
    '''
    The SVBRDF class represents a spatially-varying BRDF for a specific material.
    '''

    def __init__(self, depth: int, parameters: Tensor = None) -> None:
        '''
        Constructs a new SVBRDF instance with the given parameter depth and, optionally, set of parameters.

        Args:
            depth: The expected size of the spatial parameters in this SVBRDF.
            parameters: Tensor of parameters to be associated with this SVBRDF.
        '''
        self._depth = depth
        self._parameters = parameters if parameters is not None else Tensor()

    @property
    def depth(self) -> int:
        '''
        Returns the expected size of each spatial paramater in this SVBRDF.

        Returns:
            The expected value of `self.parameters.size(3)`.
        '''
        return self._depth

    @property
    def parameters(self) -> Tensor:
        '''
        Returns the parameters of this SVBRDF.

        Returns:
            The spatially-varying parameters of this SVBRDF.
        '''
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Tensor) -> None:
        '''
        Sets the parameters of this SVBRDF to the given parameter Tensor.

        Args:
            parameters: Tensor [B, R, C, X] of SVBRDF parameters.
        '''
        assert len(parameters.shape) == 4, 'SVBRDF parameters must have 4 dimensions.'
        assert parameters.size(3) == self.depth, f'SVBRDF must have {self.depth} values for each spatial parameter.'
        self._parameters = parameters

    @abc.abstractmethod
    def clone(self) -> SVBRDF:
        '''
        Returns a deep copy of this SVBRDF.

        Returns:
            Clone of this SVBRDF.
        '''
        raise NotImplementedError('Class "SVBRDF" does not implement the "clone" function.')

    def __call__(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''
        Evaluates this SVBRDF with respect to the given surface normals for each incident and outbound direction.

        Args:
            normals: Tensor [R, C, D, 3] of surface normals.
            incident_directions: Tensor [R, C, D, 3] of incident directions.
            outbound_directions: Tensor [R, C, D, 3] of outbound directions.

        Returns:
            Tensor [B, R, C, D, 3] of SVBRDF values for each incident-outbound direction pair.
        '''
        def expand_to_batch(directions: Tensor) -> Tensor:
            '''Adds a batch dimension to the given Tensor [R, C, D, 3] to match the SVBRDF parameter dimensions.'''
            return directions.unsqueeze(0).expand(self.parameters.size(0), -1, -1, -1, -1)
        return self.evaluate(incident_directions=expand_to_batch(incident_directions), normals=expand_to_batch(normals),
                             outbound_directions=expand_to_batch(outbound_directions))

    @abc.abstractmethod
    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''
        Evaluates this SVBRDF with respect to the given batch of surface normals for each incident and outbound direction.

        Args:
            normals: Tensor [B, R, C, D, 3] of surface normals.
            incident_directions: Tensor [B, R, C, D, 3] of incident directions.
            outbound_directions: Tensor [B, R, C, D, 3] of outbound directions.

        Returns:
            Tensor [B, R, C, D, 3] of SVBRDF values for each incident-outbound direction pair.
        '''
        raise NotImplementedError('Class "SVBRDF" does not implement the "evaluate" method.')

    @staticmethod
    def _split_parameters(parameters: Tensor, chunks: List[int]) -> Tuple:
        '''
        Splits an unsqueezed version of the given parameter Tensor into the specified chunks along its final dimension.

        Args:
            parameters: Tensor [B, R, C, D] of parameters.
            chunks: List of split section sizes.

        Returns:
            Tuple of [B, R, C, 1, *] Tensors that arise from the splitting procedure.
        '''
        assert sum(chunks) == parameters.size(-1), 'Chunks must accumulate to the size of the final parameter dimension.'
        return parameters.unsqueeze(3).split(chunks, dim=-1)


class LambertianSVBRDF(SVBRDF):
    '''
    The LambertianSVBRDF class represents a Lambertian SVBRDF.
    '''

    def __init__(self, parameters: Tensor = None) -> None:
        '''See SVBRDF.__init__().'''
        super().__init__(depth=3, parameters=parameters)

    def clone(self) -> LambertianSVBRDF:
        '''See SVBRDF.clone()'''
        return LambertianSVBRDF(self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        return torch.unsqueeze(self.parameters / math.pi, dim=3).expand(incident_directions.shape)


class PhongSVBRDF(SVBRDF):
    '''
    The PhongSVBRDF class represents a Phong SVBRDF lobe based on the halfway vector representation.
    '''

    def __init__(self, parameters: Tensor = None) -> None:
        '''See SVBRDF.__init__().'''
        super().__init__(depth=4, parameters=parameters)

    def clone(self) -> PhongSVBRDF:
        '''See SVBRDF.clone()'''
        return PhongSVBRDF(self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        colours, glossiness = SVBRDF._split_parameters(self.parameters, [3, 1])
        # This Phong model is based on the halfway vector parameterization.
        halfways = vector.normalize(incident_directions + outbound_directions)
        # The macrosurface normal is assumed to be (0, 0, 1).
        zeniths = vector.dot(halfways, normals).clamp(0, 1)
        # The parameterization of the specular exponent is taken from the Substance Blinn-Phong shader.
        exponents = 4 * torch.pow(2, glossiness * 11)
        # The Phong normalization factor is derived in http://www.farbrausch.de/~fg/stuff/phong.pdf.
        factors = (exponents + 2) * (exponents + 4) / (8 * math.pi * (torch.pow(1 / math.sqrt(2), exponents) + exponents))
        return colours * factors * zeniths.pow(exponents)


class BlinnPhongSVBRDF(SVBRDF):
    '''
    The BlinnPhongSVBRDF class represents a Blinn-Phong SVBRDF.
    '''

    def __init__(self, parameters: Tensor = None) -> None:
        '''See SVBRDF.__init__()'''
        super().__init__(depth=8, parameters=parameters)

    def clone(self) -> BlinnPhongSVBRDF:
        '''See SVBRDF.clone()'''
        return BlinnPhongSVBRDF(self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        albedos, diffuse_parameters, specular_parameters = torch.split(self.parameters, [1, 3, 4], dim=3)
        # Extract the relative albedos of the diffuse and specular lobes.
        specular_albedos = torch.unsqueeze(albedos, dim=3)
        diffuse_albedos = 1 - specular_albedos
        # The Blinn-Phong SVBRDF is a superposition of a Lambertian and halfway Phong SVBRDF.
        diffuse_terms = LambertianSVBRDF(diffuse_parameters).evaluate(normals, incident_directions, outbound_directions)
        specular_terms = PhongSVBRDF(specular_parameters).evaluate(normals, incident_directions, outbound_directions)
        return diffuse_albedos * diffuse_terms + specular_albedos * specular_terms


class MicrofacetSVBRDF(SVBRDF):
    '''
    The MicrofacetSVBRDF class represents a perfectly specular microfacet SVBRDF lobe.
    '''

    def __init__(self,
                 D: Tuple[Callable[[Tensor, Tensor, Tensor], Tensor], List[int]],
                 F: Tuple[Callable[[Tensor, Tensor, Tensor], Tensor], List[int]],
                 G: Tuple[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor], List[int]],
                 parameters: Tensor = None) -> None:
        '''
        Constructs a new MicrofacetSVBRDF with the given D, F, and G functions, as well as an optional parameter Tensor.

        Args:
            D: Microfacet distribution function with signature D(parameters, vectors) and parameter indices.
            F: Fresnel factor function with signature F(parameters, halfway directions, incident directions) and parameter indices.
            G: Monodirectional shadow-masking function with signature G(parameters, halfway directions, directions) and parameter indices.
            parameters: See SVBRDF.__init__().
        '''
        self._D_function, self._D_indices = D
        self._F_function, self._F_indices = F
        self._G_function, self._G_indices = G
        depth = max([0] + self._D_indices + self._F_indices + self._G_indices) + 1
        super().__init__(depth=depth, parameters=parameters)

    def clone(self) -> MicrofacetSVBRDF:
        '''See SVBRDF.clone()'''
        return MicrofacetSVBRDF(D=(self._D_function, self._D_indices),
                                F=(self._F_function, self._F_indices),
                                G=(self._G_function, self._G_indices),
                                parameters=self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        D_parameters = self.parameters[:, :, :, self._D_indices]
        F_parameters = self.parameters[:, :, :, self._F_indices]
        G_parameters = self.parameters[:, :, :, self._G_indices]
        # The halfway directions parameterize each of the D, F, and G functions so it's worth computing only once.
        halfways = vector.normalize(incident_directions + outbound_directions)
        # Evaluate the remainder of the microfacet SVBRDF in the usual way, taking into account that the value returned
        # by G is already divided by a factor of the denominator in the original microfacet model.
        D_terms = self._D_function(D_parameters, normals, halfways)
        F_terms = self._F_function(F_parameters, halfways, incident_directions)
        G_terms = self._G_function(G_parameters, normals, incident_directions, halfways) * \
                  self._G_function(G_parameters, normals, outbound_directions, halfways)
        return F_terms * D_terms * G_terms

    @staticmethod
    def D_Berry(parameters: Tensor, normals: Tensor, directions: Tensor) -> Tensor:
        '''
        Evaluates the Berry normal distribution (GTR with γ=1) for the provided normals and directions using the given parameters.
        See https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, 1] of α parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of directions.

        Returns:
            Tensor [B, R, C, D, 3] of normal distribution function values at each vector.
        '''
        assert parameters.size(3) == 1, 'Berry normal distribution must have 1 value for each spatial parameter.'
        alphas = torch.unsqueeze(parameters, dim=3)
        zeniths = vector.dot(directions, normals)
        return (alphas**2 - 1) / (math.pi * torch.log(alphas**2) * (1 + (alphas**2 - 1) * zeniths**2))

    @staticmethod
    def D_GGX_Anisotropic(parameters: Tensor, normals: Tensor, directions: Tensor) -> Tensor:
        '''
        Evaluates the anisotropic GGX normal distribution for the provided normals and directions using the given parameters.
        See https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, 3] of α[x], α[y], and anisotropy angle parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of directions.

        Returns:
            Tensor [B, R, C, D, 3] of normal distribution function values at each vector.
        '''
        assert parameters.size(3) == 3, 'Anisotropic GGX normal distribution must have 3 values for each spatial parameter.'
        alphas_x, alphas_y, anisotropy_angles = SVBRDF._split_parameters(parameters, [1, 1, 1])
        # Rotate the directions counter-clockwise by their respective anisotropy angles.
        rotated_directions = torch.cat([torch.cos(anisotropy_angles) * directions[:, :, :, :, [0]] - torch.sin(anisotropy_angles) * directions[:, :, :, :, [1]],
                                        torch.sin(anisotropy_angles) * directions[:, :, :, :, [0]] + torch.cos(anisotropy_angles) * directions[:, :, :, :, [1]],
                                        directions[:, :, :, :, [2]]], dim=4)
        # Scale the directions by their reciprocated alpha parameters (this is just a computational convenience).
        roughened_directions = torch.cat([rotated_directions[:, :, :, :, [0]] / alphas_x,
                                          rotated_directions[:, :, :, :, [1]] / alphas_y,
                                          rotated_directions[:, :, :, :, [2]]], dim=4)
        # Technically, the anisotropic GGX distribution should be clamped below the upper hemisphere; however, the upper
        # hemisphere can be flipped if it is convenient to do so (by negating the normal) and so this fact is ignored.
        return 1 / (math.pi * alphas_x * alphas_y * vector.dot(roughened_directions, roughened_directions)**2)

    @staticmethod
    def D_GGX_Isotropic(parameters: Tensor, normals: Tensor, directions: Tensor) -> Tensor:
        '''
        Evaluates the isotropic GGX normal distribution for the provided normalas and directions using the given parameters.
        See https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, 1] of α parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of directions.

        Returns:
            Tensor [B, R, C, D, 3] of normal distribution function values at each vector.
        '''
        assert parameters.size(3) == 1, 'Isotropic GGX normal distribution must have 1 value for each spatial parameter.'
        alphas = torch.unsqueeze(parameters, dim=3)
        zeniths = vector.dot(directions, normals)
        return alphas**2 / (math.pi * (1 + (alphas**2 - 1) * zeniths**2)**2)

    @staticmethod
    def F_Disney(parameters: Tensor, normals: Tensor, incident_directions: Tensor) -> Tensor:
        '''
        Computes the Disney Fresnel approximation for the provided normals and incident directions using the given parameters.
        See https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, D, 1] of grazing colour parameters.
            normals: Tensor [B, R, C, D, 3] of interface normals.
            incident_directions: Tensor [B, R, C, D, 3] of incident (or outbound) directions.

        Returns:
            Tensor [B, R, C, D, 3] of Fresnel values for each incident direction.
        '''
        assert parameters.size(3) == 1, 'Disney Fresnel model must have 1 value for each incident direction.'
        grazing_colours = parameters
        difference_angles = vector.dot(normals, incident_directions)
        return 1 + (grazing_colours - 1) * (1 - difference_angles)**5

    @staticmethod
    def F_Schlick(parameters: Tensor, normals: Tensor, incident_directions: Tensor) -> Tensor:
        '''
        Computes the Schlick Fresnel approximation for the provided normals and incident directions using the given parameters.
        See https://en.wikipedia.org/wiki/Schlick%27s_approximation for more details.

        Args:
            parameters: Tensor [B, R, C, 3] of reflection colour parameters.
            normals: Tensor [B, R, C, D, 3] of interface normals.
            incident_directions: Tensor [B, R, C, D, 3] of incident (or outbound) directions.

        Returns:
            Tensor [B, R, C, D, 3] of Fresnel values for each incident direction.
        '''
        assert parameters.size(3) == 3, 'Schlick F approximation must have 3 values for each spatial parameter.'
        reflection_colours = torch.unsqueeze(parameters, dim=3)
        difference_angles = vector.dot(normals, incident_directions)
        return reflection_colours + (1 - reflection_colours) * (1 - difference_angles)**5

    @staticmethod
    def F_Schlick_SG(parameters: Tensor, normals: Tensor, incident_directions: Tensor) -> Tensor:
        '''
        Computes a spherical Gaussian approximation of the Schlick Fresnel approximation for the provided normals and
        incident directions using the given parameters.  For more details, see
        https://seblagarde.wordpress.com/2012/06/03/spherical-gaussien-approximation-for-blinn-phong-phong-and-fresnel/.

        Args:
            parameters: Tensor [B, R, C, 3] of reflection colour parameters.
            normals: Tensor [B, R, C, D, 3] of interface directions.
            incident_directions: Tensor [B, R, C, D, 3] of incident (or outbound) directions.

        Returns:
            Tensor [B, R, C, D, 3] of Fresnel values for each incident direction.
        '''
        assert parameters.size(3) == 3, 'Schlick F approximation with SG must have 3 values for each spatial parameter.'
        reflection_colours = torch.unsqueeze(parameters, dim=3)
        difference_angles = vector.dot(normals, incident_directions)
        return reflection_colours + (1 - reflection_colours) * torch.pow(2, (-5.55473 * difference_angles - 6.98316) * difference_angles)

    @staticmethod
    def G_GGX_Anisotropic(parameters: Tensor, normals: Tensor, directions: Tensor, halfways: Tensor) -> Tensor:
        '''
        Evaluates the anisotropic GGX shadow-masking G1 function for the provided halfway and (incident or outbound) directions
        using the given parameters.  See http://jcgt.org/published/0003/02/03/paper.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, 3] of α[x], α[y], and anisotropy angle parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of (incident or outbound) directions.
            halfways: Tensor [B, R, C, D, 3] of halfway directions.

        Returns:
            Tensor [B, R, C, D, 3] of G1 visibility values for each (incident or outbound) direction.
        '''
        assert parameters.size(3) == 3, 'Anisotropic GGX shadow-masking G1 function must have 3 values for each spatial parameter.'
        alphas_x, alphas_y, anisotropy_angles = SVBRDF._split_parameters(parameters, [1, 1, 1])
        # Rotate the directions counter-clockwise by their respective anisotropy angles.
        rotated_directions = torch.cat([torch.cos(anisotropy_angles) * directions[:, :, :, :, [0]] - torch.sin(anisotropy_angles) * directions[:, :, :, :, [1]],
                                        torch.sin(anisotropy_angles) * directions[:, :, :, :, [0]] + torch.cos(anisotropy_angles) * directions[:, :, :, :, [1]],
                                        directions[:, :, :, :, [2]]], dim=4)
        # Scale the directions by their alpha parameters (this is just a computational convenience).
        roughened_directions = torch.cat([rotated_directions[:, :, :, :, [0]] * alphas_x,
                                          rotated_directions[:, :, :, :, [1]] * alphas_y,
                                          rotated_directions[:, :, :, :, [2]]], dim=4)
        # Microfacets are not visible from behind; the active side of a microfacet is determined by the surface normal.
        visible = torch.sign(vector.dot(directions, halfways) * vector.dot(directions, normals)) == 1
        # The returned value is divided by 2 * |normals ∙ directions|.
        return visible / (roughened_directions[:, :, :, :, [2]] + torch.sqrt(vector.dot(roughened_directions, roughened_directions)))

    @staticmethod
    def G_GGX_Isotropic(parameters: Tensor, normals: Tensor, directions: Tensor, halfways: Tensor) -> Tensor:
        '''
        Evaluates the isotropic GGX shadow-masking G1 function for the provided halfway and (incident or outbound) directions
        using the given parameters.  See https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf for more details.

        Args:
            parameters: Tensor [B, R, C, 4] of α and normal parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of (incident or outbound) directions.
            halfways: Tensor [B, R, C, D, 3] of halfway directions.

        Returns:
            Tensor [B, R, C, D, 3] of G1 visibility values for each (incident or outbound) direction.
        '''
        assert parameters.size(3) == 1, 'Isotropic GGX shadow-masking G1 function must have 1 value for each spatial parameter.'
        alphas = torch.unsqueeze(parameters, dim=3)
        visible = torch.sign(vector.dot(directions, halfways) * vector.dot(directions, normals)) == 1
        zeniths = vector.dot(directions, normals)
        return visible / (zeniths + torch.sqrt(alphas**2 + (1 - alphas**2) * zeniths**2))

    @staticmethod
    def G_Schlick(parameters: Tensor, normals: Tensor, directions: Tensor, halfways: Tensor) -> Tensor:
        '''
        Computes the Schlick shadow-masking approximation for the provided halfway and (incident or outbound) directions
        using the given parameters.  See https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        for more details.

        Args:
            parameters: Tensor [B, R, C, 4] of α and normal parameters.
            normals: Tensor [B, R, C, D, 3] of surface normals.
            directions: Tensor [B, R, C, D, 3] of (incident or outbound) directions.
            halfways: Tensor [B, R, C, D, 3] of halfway directions.

        Returns:
            Tensor [B, R, C, D, 3] of G1 visibility values for each (incident or outbound) direction.
        '''
        assert parameters.size(3) == 1, 'Schlick shadow-masking G1 function must have 1 value for each spatial parameter.'
        alphas = torch.unsqueeze(parameters, dim=3)
        zeniths = vector.dot(directions, normals)
        return 1 / (zeniths * (2 - alphas) + alphas)


class SubstanceSVBRDF(SVBRDF):
    '''
    The SubstanceSVBRDF class represents the physically_specular_glossiness SVBRDF in Substance Designer.
    '''

    def __init__(self, parameters: Tensor = None) -> None:
        '''See SVBRDF.__init__()'''
        super().__init__(depth=9, parameters=parameters)

    def clone(self) -> SubstanceSVBRDF:
        '''See SVBRDF.clone()'''
        return SubstanceSVBRDF(self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        diffuse_colour, specular_colour, glossiness, anisotropy_levels, anisotropy_angles = SVBRDF._split_parameters(self.parameters, [3, 3, 1, 1, 1])
        # Normalize the diffuse colour to ensure the SVBRDF conserves energy.
        diffuse_parameters = diffuse_colour * (1 - specular_colour)
        # Roughness (√α) is defined by the glossiness and anisotropy level parameters.
        roughness_x = 1 - glossiness.clamp(1E-5, 1 - 1E-5)
        roughness_y = roughness_x / torch.sqrt(1 - anisotropy_levels.clamp(0, 1 - 1E-5))
        # The order of the concatenation affects the indices passed to the MicrofacetSVBRDF constructor.
        specular_parameters = torch.cat([roughness_x**2, roughness_y**2, 2 * math.pi * anisotropy_angles, specular_colour], dim=4)
        # The Blinn-Phong SVBRDF is a superposition of a Lambertian and a Microfacet SVBRDF.
        diffuse_terms = LambertianSVBRDF(diffuse_parameters.squeeze(dim=3)).evaluate(normals, incident_directions, outbound_directions)
        specular_terms = MicrofacetSVBRDF(D=(MicrofacetSVBRDF.D_GGX_Anisotropic, [0, 1, 2]),
                                          F=(MicrofacetSVBRDF.F_Schlick_SG, [3, 4, 5]),
                                          G=(MicrofacetSVBRDF.G_GGX_Anisotropic, [0, 1, 2]),
                                          parameters=specular_parameters.squeeze(dim=3)).evaluate(normals, incident_directions, outbound_directions)
        return diffuse_terms + specular_terms


class DisneySVBRDF(SVBRDF):
    '''
    The DisneySVBRDF class represents the Disney SVBRDF.
    '''

    def __init__(self, parameters: Tensor = None) -> None:
        '''See SVBRDF.__init__()'''
        super().__init__(depth=14, parameters=parameters)

    def clone(self) -> DisneySVBRDF:
        '''See SVBRDF.clone()'''
        return DisneySVBRDF(self.parameters.clone().detach())

    def evaluate(self, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor) -> Tensor:
        '''See SVBRDF.evaluate().'''
        base_colours, subsurface, metallic, specular_amounts, specular_tints, roughness, anisotropy_levels, anisotropy_angles, \
            sheen_amounts, sheen_tints, clearcoat_amounts, clearcoat_gloss = torch.split(self.parameters, [3] + [1] * 11, dim=-1)

        # The incident and outbound halfway zeniths are shared across several SVBRDF lobes.
        halfways = vector.normalize(incident_directions + outbound_directions)
        halfway_zeniths = vector.dot(incident_directions, halfways)

        # Isolate the hue and saturation from the base colour luminosity: https://www.w3.org/Graphics/Color/sRGB.
        dtype = self.parameters.dtype
        device = utils.get_device_name()
        luminosity = torch.sum(1E-5 + base_colours * torch.tensor([0.2126, 0.7152, 0.0722], dtype=dtype, device=device), dim=-1, keepdim=True)
        tint_colours = base_colours / luminosity

        # Compute the Lambertian SVBRDF contribution with two Fresnel factors.
        diffuse_lambertian_colours = 0.5 + 2 * torch.unsqueeze(roughness, dim=3) * halfway_zeniths**2
        diffuse_lambertian_fresnel_factors = MicrofacetSVBRDF.F_Disney(diffuse_lambertian_colours, normals, incident_directions) * \
                                             MicrofacetSVBRDF.F_Disney(diffuse_lambertian_colours, normals, outbound_directions)
        diffuse_lambertian_terms = LambertianSVBRDF(base_colours).evaluate(normals, incident_directions, outbound_directions) \
                                   * diffuse_lambertian_fresnel_factors

        incident_zeniths = vector.dot(incident_directions, normals).clamp(1E-5, 1)
        outbound_zeniths = vector.dot(outbound_directions, normals).clamp(1E-5, 1)

        # Compute the Hanrahan-Krueger-inspired SVBRDF contribution (to model subsurface scattering).
        diffuse_subsurface_colours = torch.unsqueeze(roughness, dim=3) * halfway_zeniths**2
        diffuse_subsurface_fresnel_factors = MicrofacetSVBRDF.F_Disney(diffuse_subsurface_colours, normals, incident_directions) * \
                                             MicrofacetSVBRDF.F_Disney(diffuse_subsurface_colours, normals, outbound_directions)
        diffuse_subsurface_terms = LambertianSVBRDF(base_colours).evaluate(normals, incident_directions, outbound_directions) \
                                   * 1.25 * (diffuse_subsurface_fresnel_factors * (1 / (incident_zeniths + outbound_zeniths) - 0.5) + 0.5)

        # Compute the sheen SVBRDF contribution (for cloth materials).
        diffuse_sheen_colours = torch.lerp(torch.tensor(1.0, dtype=dtype, device=device), tint_colours, sheen_tints)
        diffuse_sheen_terms = torch.unsqueeze(sheen_amounts * diffuse_sheen_colours, dim=3) * (1 - halfway_zeniths)**5

        # Compute the primary specular SVBRDF contribution using an anisotropic GGX microfacet model.
        specular_aspects = torch.sqrt(1.0 - 0.9 * anisotropy_levels)
        specular_alphas_x = torch.max(roughness**2 / specular_aspects, torch.tensor([1E-3], dtype=dtype, device=device))
        specular_alphas_y = torch.max(roughness**2 * specular_aspects, torch.tensor([1E-3], dtype=dtype, device=device))
        specular_colours = torch.lerp(specular_amounts * 0.08 * torch.lerp(torch.tensor(1.0, dtype=dtype, device=device), tint_colours, specular_tints), base_colours, metallic)
        specular_parameters = torch.cat([specular_alphas_x, specular_alphas_y, 2 * math.pi * anisotropy_angles, specular_colours], dim=3)
        specular_terms = MicrofacetSVBRDF(D=(MicrofacetSVBRDF.D_GGX_Anisotropic, [0, 1, 2]),
                                          F=(MicrofacetSVBRDF.F_Schlick, [3, 4, 5]),
                                          G=(MicrofacetSVBRDF.G_GGX_Anisotropic, [0, 1, 2]),
                                          parameters=specular_parameters).evaluate(normals, incident_directions, outbound_directions)

        # Compute the clearcoat SVBRDF contribution using a GTR model where γ=1.
        clearcoat_D_alphas = torch.lerp(torch.full(clearcoat_gloss.shape, 0.1, dtype=dtype, device=device), torch.full(clearcoat_gloss.shape, 0.001, dtype=dtype, device=device), clearcoat_gloss)
        clearcoat_G_alphas = torch.full(clearcoat_D_alphas.shape, 0.25, dtype=dtype, device=device)
        clearcoat_colours = torch.full(base_colours.shape, 0.04, dtype=dtype, device=device)
        clearcoat_parameters = torch.cat([clearcoat_D_alphas, clearcoat_G_alphas, clearcoat_colours], dim=3)
        clearcoat_terms = MicrofacetSVBRDF(D=(MicrofacetSVBRDF.D_Berry, [0]),
                                           F=(MicrofacetSVBRDF.F_Schlick, [2, 3, 4]),
                                           G=(MicrofacetSVBRDF.G_GGX_Isotropic, [1]),
                                           parameters=clearcoat_parameters).evaluate(normals, incident_directions, outbound_directions)

        # Cull any reflections that pass through the surface.
        visible = (incident_zeniths >= 0) & (outbound_zeniths >= 0)
        # Combine the SVBRDF contributions in a principled way.
        diffuse_terms = diffuse_sheen_terms + torch.lerp(diffuse_lambertian_terms, diffuse_subsurface_terms, torch.unsqueeze(subsurface, dim=3))
        combined_terms = (1 - torch.unsqueeze(metallic, dim=3)) * diffuse_terms + specular_terms + torch.unsqueeze(clearcoat_amounts, dim=3) * 0.25 * clearcoat_terms
        return visible * combined_terms
