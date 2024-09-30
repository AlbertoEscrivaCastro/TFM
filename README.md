# Detección de Anomalías en Telemetría de Satélites mediante IA Generativa y Agentes Autónomos

Autor: Alberto Escrivá Castro

### - DEPENDENCIAS -
https://github.com/Jhryu30/AnomalyBERT    -->  Este repositorio debe ponerse dentro de este proyecto para que funcione, quedando: ./TFM/AnomalyBERT/


### - VERSIONES -
Por temas de compatibilidad se han empleado:
    - python 3.8
    - torch 2.2.0

También se ha tenido que realizar el siguietne ajuste en las librerías:
En ".\Python\Python38\site-packages\torch\nn\modules\activation.py" reemplazar la clase GELU por la verisón anterior mostrada a continuación ( https://github.com/czg1225/SlimSAM/issues/1 ):

    class GELU(Module):
        r"""Applies the Gaussian Error Linear Units function:

        .. math:: \text{GELU}(x) = x * \Phi(x)

        where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

        Shape:
            - Input: :math:`(N, *)` where `*` means, any number of additional
              dimensions
            - Output: :math:`(N, *)`, same shape as the input

        .. image:: ../scripts/activation_images/GELU.png

        Examples::

            >>> m = nn.GELU()
            >>> input = torch.randn(2)
            >>> output = m(input)
        """
        def forward(self, input: Tensor) -> Tensor:
            return F.gelu(input)
