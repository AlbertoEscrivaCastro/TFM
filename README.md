# Detección de Anomalías en Telemetría de Satélites mediante IA Generativa y Agentes Autónomos

Autor: Alberto Escrivá Castro

### - DEPENDENCIAS -
https://github.com/Jhryu30/AnomalyBERT    -->  Este repositorio debe ponerse dentro de este proyecto para que funcione, quedando: ./TFM/AnomalyBERT/


### - VERSIONES -
Se han seguido las instrucciones de instalación del entorno, así como versiones y requerimientos de https://github.com/Jhryu30/AnomalyBERT a excepción de:

    Python 3.8.18
    PyTortch 2.2.0+cu118

También se ha tenido que realizar el siguiente ajuste en las librerías:
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

De este modo se compatibiliza AnomalyBERT con los Transformer Agents.
