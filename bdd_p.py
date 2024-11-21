#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
from copy import deepcopy, copy
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyreadstat import read_sav, write_sav, metadata_container


# Versión 3.03


class ErrBDD(Exception):
    pass


class BaseDatos:
    """
    Crea un objeto BaseDatos sin datos y sin necesidad de incluir un archivo
    """

    def __init__(self, ruta: str = None, encoding: str = None, nombre: str = None, _imp: bool = True) -> None:
        if ruta is None and nombre is None:
            raise ErrBDD("Ambos, [nombre] y [ruta] no pueden estar vacíos")

        self.vars_base = []

        self.eti_vars = {}
        self.eti_values = {}
        self.medidas_vars = {}
        self.formatos_vars = {}

        if ruta is not None:
            self.ruta = ruta
            if nombre is None:
                nombre = os.path.split(self.ruta)[1].split('.')[0]
            if encoding is None:
                # También puede ser "L1", "latin1" o "ISO-8859-1", los tres para iso 8859-1
                encoding = 'UTF-8'
            self.df, self._metadata = read_sav(ruta, encoding=encoding)

            # Variables de la base
            self.vars_base.extend(self._metadata.column_names)

            # Etiqueta de las variables
            self.eti_vars.update(
                {var: var for var in self.vars_base})  # Si no tiene etiqueta se pone el nombre de la variable
            self.eti_vars.update(self._metadata.column_names_to_labels)

            # Etiquetas de los códigos
            self.eti_values.update({var: {} for var in self.vars_base})
            self.eti_values.update(self._metadata.variable_value_labels)

            # Tamaño de las variables
            self.medidas_vars.update(self._metadata.variable_measure)

            # Formato de las variables
            self.formatos_vars.update(self._metadata.original_variable_types)
        else:
            self.df = pd.DataFrame()
            self._metadata = metadata_container

        self.nombre = nombre
        self._atribs_var = (self.eti_vars, self.eti_values, self.medidas_vars, self.formatos_vars)

        if _imp:
            print(f'Base [{self.nombre}] creada.')

    def __str__(self):
        return self.df.to_string()

    @property
    def atribs_var(self):
        return self._atribs_var

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, nom):
        self._nombre = nom

    def vars(self, *rangos) -> list:
        """
        Devuelve una lista con las variables en los rangos proporcionados
        Si no se incluyen rangos se devuleve la lista completa de variables
        """

        vars_base_ = self.vars_base

        if not rangos:
            return vars_base_

        l_vars = []

        for rango in rangos:
            if isinstance(rango, str):
                if rango not in vars_base_:
                    raise ErrBDD(f'No existe [{rango}] en la base de datos.')
                l_vars.append(rango)
            else:
                msj_err = None
                ind_ini, ind_fin = None, None
                try:
                    ind_ini, ind_fin = tuple(vars_base_.index(nvar) for nvar in rango)
                except ValueError as e:
                    msj_err = str(e)
                else:
                    if ind_ini >= ind_fin:
                        msj_err = f'Error en orden de rango de variables: [{rango}]'

                if msj_err:
                    raise ErrBDD(f'Grupo mal especificado: {msj_err}.')

                l_vars.extend(vars_base_[ind_ini: ind_fin + 1])

        duplicados = [item for item, count in Counter(l_vars).items() if count > 1]

        if duplicados:
            raise ErrBDD(f'Variables duplicadas: {tuple(duplicados)}.')

        return l_vars

    def get_var_eti(self, l_var: [str, list, tuple]) -> [str, dict]:
        """
        Regresa la etiqueta de una variable
        """
        if isinstance(l_var, str):
            return self.eti_vars[self.vars(l_var)[0]]
        else:
            return {vr: self.eti_vars[vr] for vr in self.vars(l_var)}

    def set_var_eti(self, l_var: [str, list, tuple], etiqueta: [str, callable]) -> None:
        """
        Modifica la etiqueta de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiqueta):
            callback = etiqueta
        else:
            def callback(*args):
                return etiqueta

        for ind, nvar in enumerate(l_var):
            eti_o = self.eti_vars[nvar]
            self.eti_vars[nvar] = callback(nvar, eti_o, ind)

    def get_val_eti(self, l_var: [str, list, tuple]) -> dict:
        """
        Regresa las etiquetas de los valores de una variable
        """
        if isinstance(l_var, str):
            return self.eti_values[self.vars(l_var)[0]]
        else:
            return {vr: self.eti_values[vr] for vr in self.vars(l_var)}

    def set_val_eti(self, l_var: [str, list, tuple], etiquetas: [dict[[int, float], str], callable]) -> None:
        """
        Modifica las etiquetas de los valores de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiquetas):
            callback = etiquetas
        else:
            def callback(*args):
                return etiquetas

        for ind, nvar in enumerate(l_var):
            etis_o = self.eti_values[nvar]
            self.eti_values[nvar] = callback(nvar, etis_o, ind)

    def get_frmt(self, l_var: [str, list, tuple]) -> [str, dict]:
        """
        Regresa el formato de una variable
        """
        if isinstance(l_var, str):
            return self.formatos_vars[self.vars(l_var)[0]]
        else:
            return {vr: self.formatos_vars[vr] for vr in self.vars(l_var)}

    def set_frmt(self, l_var: [str, list, tuple], formatos: [str, callable]) -> None:
        """
        Mofica el formato de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(formatos):
            callback = formatos
        else:
            def callback(*args):
                return formatos

        for ind, nvar in enumerate(l_var):
            frms = self.formatos_vars[nvar]
            self.formatos_vars[nvar] = callback(nvar, frms, ind)

    def renombrar_var(self, l_var: [str, list, tuple], nombre: [str, callable]) -> None:
        """
        Renombra una variable en la metadata y el DataFrame
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            if not len(l_var) == 0:
                l_var = self.vars(*l_var)

        if callable(nombre):
            callback = nombre
        else:
            def callback(*args):
                return nombre

        for ind, nvar in enumerate(l_var):
            ind_var = self.vars_base.index(nvar)
            eti_o = self.eti_vars[nvar]
            nuevo_nvar = callback(nvar, eti_o, ind)

            self.vars_base[ind_var] = nuevo_nvar

            # Modificar metadatos
            # eti_vars, eti_values, medidas_vars, formatos_vars
            # for self_dic in (self.eti_vars, self.eti_values, self.medidas_vars, self.formatos_vars):
            for self_dic in self.atribs_var:
                self_dic[nuevo_nvar] = self_dic.pop(nvar)

            self.df.rename(columns={nvar: nuevo_nvar}, inplace=True)

        duplicados = [elem for elem, cuenta in Counter(self.vars_base).items() if cuenta > 1]

        if duplicados:
            raise ErrBDD(f'Variables duplicadas: [{tuple(duplicados)}].')

    def incluir_vars(self, l_var: [str, list, tuple], elim: bool = False, orden_base: bool = False,
                     todas: bool = False) -> None:
        """
        Solo deja las variables listadas
        """
        if elim and (todas or len(l_var) == 0):
            raise ErrBDD('Se eliminarían todas las variables de la base.')

        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if elim:
            l_var = [nv for nv in self.vars_base if nv not in l_var]

        if orden_base:
            l_var = [nv for nv in self.vars_base if nv in l_var]

        if todas:
            l_var += [nv for nv in self.vars_base if nv not in l_var]

        # Modificar metadatos
        # Aquí sólo se quitan las variables de los atributos
        # eti_vars, eti_values, medidas_vars, formatos_vars
        for self_dic in self.atribs_var:
            [self_dic.pop(var, None) for var in self.vars_base if var not in l_var]

        self.df = self.df.loc[:, l_var]
        self.df = self.df.copy()

        self.vars_base = l_var

    def borrar_vars(self, l_var: [str, list, tuple]) -> None:
        """
        Elimina las variables listadas
        """
        if not len(l_var) == 0:
            self.incluir_vars(l_var, elim=True)

    def crear_var(self, nombre: str, antes: str = None, despues: str = None, medida: str = 'nominal', eti: str = None,
                  fmt: str = 'F3', eti_vals: dict[[int, float], str] = None) -> None:
        """
        Crea una nueva variable
        """
        if nombre in self.vars_base:
            raise ErrBDD(f'Ya existe la variable [{nombre}]')

        if antes and despues:
            raise ErrBDD('Sólo debe especificarse un parámetro dentro de [antes, despues].')
        elif antes or despues:
            err_msj = None
            if antes is not None and antes not in self.vars_base:
                err_msj = antes
            if despues is not None and despues not in self.vars_base:
                err_msj = despues

            if err_msj is not None:
                raise ErrBDD(f'No existe la variable [{err_msj}].')
            pos = antes or despues
        else:
            if len(self.vars_base) != 0:
                pos = self.vars_base[-1]
            else:
                pos = None

        # Índice de inserción
        if pos is not None:
            indice = self.vars_base.index(pos)
        else:
            indice = -1
        if despues or not antes:
            indice += 1

        # Agregar en datos
        dato = '' if 'A' in fmt else np.nan
        self.df = self.df.copy()
        self.df.insert(indice, nombre, dato)

        # eti_vars, eti_values, medidas_vars, formatos_vars
        self.eti_vars[nombre] = eti if eti is not None else nombre
        self.eti_values[nombre] = eti_vals if eti_vals is not None else {}
        self.medidas_vars[nombre] = medida
        self.formatos_vars[nombre] = fmt

        # Agregar en variables
        self.vars_base.insert(indice, nombre)
        # self.vars_base = self.vars_base.copy()

    def comput_var(self, nombre: str, valor, crear: bool = False) -> None:
        """
        Calcula un valor y lo asigna a la variable
        """
        if crear:
            self.crear_var(nombre)

        if nombre not in self.vars_base:
            raise ErrBDD(f'No existe la variable [{nombre}].')

        if callable(valor):
            callback = valor
        else:
            def callback(r):
                return valor

        self.df.loc[:, nombre] = self.df.apply(callback, axis=1)

    def recod_var(self, nombre: [str, list, tuple], rec: [dict[[int, float], [int, float]], callable]) -> None:
        """
        Recodifica una variable
        """
        if isinstance(nombre, str):
            nombres = self.vars(nombre)
        else:
            nombres = self.vars(*nombre)

        if callable(rec):
            callback = rec
        else:
            def callback(r):
                return rec

        listas = zip(*callback(self.df).items())
        df_tmp = self.df.loc[:, nombres]
        df_tmp.replace(*listas, inplace=True)

        self.df.loc[:, nombres] = df_tmp

    def guardar_bdd(self, ruta: str, forzar: bool = False) -> None:
        """
        Guarda la base de datos en disco
        """
        if not forzar:
            if Path(ruta).is_file():
                raise ErrBDD(f"El archivo '{ruta}' ya existe.")
        else:
            try:
                with open(ruta, 'w') as _:
                    pass
            except PermissionError:
                raise ErrBDD(f"El archivo '{ruta}' esta siendo usado por otro programa, no se puede guardar.")

        write_sav(self.df, ruta, column_labels=self.eti_vars, variable_value_labels=self.eti_values,
                  variable_measure=self.medidas_vars, variable_format=self.formatos_vars, compress=True)
        """
        Parámetros que se le pueden pasar a write_sav
        df : pandas data frame
            pandas data frame to write to sav or zsav
        dst_path : str or pathlib.Path
            full path to the result sav or zsav file
        file_label : str, optional
            a label for the file
        column_labels : list or dict, optional
            labels for columns (variables), if list must be the same length as the number of columns. Variables with no
            labels must be represented by None. If dict values must be variable names and values variable labels.
            In such case there is no need to include all variables; labels for non existent
            variables will be ignored with no warning or error.
        compress : boolean, optional
            if true a zsav will be written, by default False, a sav is written
        row_compress : boolean, optional
            if true it applies row compression, by default False, compress and row_compress cannot be both true at the
            same time
        note : str, optional
            a note to add to the file
        variable_value_labels : dict, optional
            value labels, a dictionary with key variable name and value a dictionary with key values and
            values labels. Variable names must match variable names in the dataframe otherwise will be
            ignored. Value types must match the type of the column in the dataframe.
        missing_ranges : dict, optional
            user defined missing values. Must be a dictionary with keys as variable names matching variable
            names in the dataframe. The values must be a list. Each element in that list can either be
            either a discrete numeric or string value (max 3 per variable) or a dictionary with keys 'hi' and 'lo' to
            indicate the upper and lower range for numeric values (max 1 range value + 1 discrete value per
            variable). hi and lo may also be the same value in which case it will be interpreted as a discrete
            missing value.
            For this to be effective, values in the dataframe must be the same as reported here and not NaN.
        variable_display_width : dict, optional
            set the display width for variables. Must be a dictonary with keys being variable names and
            values being integers.
        variable_measure: dict, optional
            sets the measure type for a variable. Must be a dictionary with keys being variable names and
            values being strings one of "nominal", "ordinal", "scale" or "unknown" (default).
        variable_format: dict, optional
            sets the format of a variable. Must be a dictionary with keys being the variable names and 
            values being strings defining the format. See README, setting variable formats section,
            for more information.
        """

        print(f'Guardada en [{ruta}].')

    def juntar_horizontal(self, bd, identificador: [str, list, tuple], nombre: str = None, imp: bool = True,
                          mantener_metadata: str = 'ambos', suf_1: str = '1', suf_2: str = '2'):
        bd_ret = agregar_variables(self, bd, identificador=identificador, nombre=nombre, imp=imp,
                                   mantener=mantener_metadata, suf_1=suf_1, suf_2=suf_2)
        self.df = bd_ret.df
        for atr_self, atr_ret in zip(self.atribs_var, bd_ret.atribs_var):
            atr_self.clear()
            atr_self.update(atr_ret)

    def juntar_vertical(self, bd, identificador: [str, list, tuple], verify_integrity: bool = True, nombre: str = None,
                        imp: bool = True, mantener_metadata: str = 'arr', suf_1: str = '1', suf_2: str = '2'):
        bd_ret = agregar_casos(self, bd, identificador=identificador, verify_integrity=verify_integrity, nombre=nombre,
                               imp=imp, mantener=mantener_metadata, suf_1=suf_1, suf_2=suf_2)
        self.df = bd_ret.df
        for atr_self, atr_ret in zip(self.atribs_var, bd_ret.atribs_var):
            atr_self.clear()
            atr_self.update(atr_ret)


def juntar_metadata(bd_1: BaseDatos, bd_2: BaseDatos, nombre: str = None, mantener: str = 'ambos', suf_1: str = '1',
                    suf_2: str = '2', ident: [str, list, tuple] = None) -> BaseDatos:
    """
    Hace una copia de objeto [bd_1], y agrega la metadata de [bd_2]. Devuelve un obejto BaseDatos con un Dataframe vacío
    mantener: 'ambos', '1', '2'.
     Con 'ambos' se mantiene la metadata de ambas bases en caso de que haya variables repetidas,
      agregando el sufijo correspondiente y un guion bajo al final del nombre de dichas variables.
     Con '1' o '2', sólo se mantiene la metadata de la base correspondiente sin usar los sufijos.
    """
    # Chequeos de bases vacías
    # if not isinstance(bd_1, BaseDatos) and not isinstance(bd_2, BaseDatos):
    #     raise ErrBDD("Se está intentando juntar la metadata de dos bases vacías.")
    assert mantener in ('ambos', '1', '2'), "Error en el parámetro [mantener]"

    if isinstance(ident, str):
        ident = [ident]

    if nombre is None:
        nombre = f'{bd_1.nombre} + {bd_2.nombre}'
    bd_ret = BaseDatos(nombre=nombre, _imp=False)

    # Se divide la lista de variables a renombrar entre las que están en [bd_1] y las que no
    # vars_ambas = [vr for vr in bd_1.vars_base if vr in bd_2.vars_base]
    vars_ambas, vars_solo_2 = [], []
    for x in bd_2.vars_base:
        (vars_solo_2, vars_ambas)[x in bd_1.vars()].append(x)
    vars_solo_1 = [vr for vr in bd_1.vars_base if vr not in vars_ambas]

    if ident is not None:
        vars_ambas = [vr for vr in vars_ambas if vr not in ident]

    if mantener == 'ambos':
        # Se copian los nombres de las variables
        bd_1.renombrar_var(vars_ambas, lambda nvar, eti_o, ind: f'{nvar}_{suf_1}')
        bd_2.renombrar_var(vars_ambas, lambda nvar, eti_o, ind: f'{nvar}_{suf_2}')
        # bd_ret.vars_base = [vr if vr not in vars_ambas else f'{pref_1}_{vr}' for vr in bd_1.vars_base] + \
        #                    [vr if vr not in vars_ambas else f'{pref_2}_{vr}' for vr in bd_2.vars_base]
        bd_ret.vars_base = bd_1.vars_base + bd_2.vars_base

        # Se copian los tipos de los DataFrame
        d_tipos = dict(copy(bd_1.df.dtypes))
        d_tipos.update(bd_2.df.dtypes)

        # Aquí se va por cada uno de los diccionarios que se ocupan al guardar la base
        for dic_ret, dic_1, dic_2 in zip(bd_ret.atribs_var, bd_1.atribs_var, bd_2.atribs_var):
            # Como no hay variables repetidas, no se reemplaza nada (excepto el identificador)
            dic_ret.update(dic_1)
            dic_ret.update(dic_2)

    elif mantener == '1':
        # Se copian los nombres de las variables
        bd_ret.vars_base = bd_1.vars_base + vars_solo_2

        # Se copian los tipos de los DataFrame
        d_tipos = copy(bd_1.df.dtypes)
        if vars_solo_2:
            d_tipos.update({vr: tp for vr, tp in bd_2.df.dtypes.items() if vr in vars_solo_2})

        # Aquí se va por cada uno de los diccionarios que se ocupan al guardar la base
        for dic_ret, dic_1, dic_2 in zip(bd_ret.atribs_var, bd_1.atribs_var, bd_2.atribs_var):
            # Los valores en dic_2 se reemplazan con los de dic_1 debido al orden
            dic_ret.update(dic_2)
            dic_ret.update(dic_1)

    elif mantener == '2':
        # Se copian los nombres de las variables
        bd_ret.vars_base = vars_solo_1 + bd_2.vars_base

        # Se copian los tipos de los DataFrame
        d_tipos = copy(bd_2.df.dtypes)
        if vars_solo_1:
            d_tipos.update({vr: tp for vr, tp in bd_1.df.dtypes.items() if vr in vars_solo_1})

        # Aquí se va por cada uno de los diccionarios que se ocupan al guardar la base
        for dic_ret, dic_1, dic_2 in zip(bd_ret.atribs_var, bd_1.atribs_var, bd_2.atribs_var):
            # Los valores en dic_1 se reemplazan con los de dic_2 debido al orden
            dic_ret.update(dic_1)
            dic_ret.update(dic_2)

    # Excepción para el identificador
    if ident is not None:
        bd_ret.vars_base = [vr for vr in bd_ret.vars_base if vr not in ident]
        bd_ret.vars_base = ident + bd_ret.vars_base

    bd_ret.df = bd_ret.df.reindex(columns=bd_ret.vars_base, fill_value=np.nan)
    # Se asignan los tipos de las columnas del DataFrame
    bd_ret.df.astype(d_tipos)

    return bd_ret


def agregar_variables(bd_izquierda: BaseDatos, bd_derecha: BaseDatos, identificador: [str, list, tuple],
                      nombre: str = None, imp: bool = True, mantener: str = 'ambos', suf_1: str = '1',
                      suf_2: str = '2') -> BaseDatos:
    """
    Junta dos bases con el mismo identificador pero con diferencias en las columnas
    """
    # Chequeos de bases
    if bd_izquierda.vars_base == bd_derecha.vars_base:
        raise ErrBDD("Ambas bases tienen las mismas variables")
    assert mantener in ('izq', 'der', 'ambos'), "Error en el parámetro [mantener_metadata]"
    _mantener_metadata = '1' if mantener == 'izq' else '2' if mantener == 'der' else mantener

    if isinstance(identificador, str):
        identificador = [identificador]
    bd_ret = juntar_metadata(bd_izquierda, bd_derecha, nombre=nombre, mantener=_mantener_metadata, suf_1=suf_1,
                             suf_2=suf_2, ident=identificador)
    # Imprime las diferencias en las variables
    vars_izq_no_der = [nv for nv in bd_izquierda.vars_base if nv not in bd_derecha.vars_base]
    if vars_izq_no_der and imp:
        print(f'\nVariables en [{bd_izquierda.nombre}] que no están en [{bd_derecha.nombre}]:')
        for v in vars_izq_no_der:
            print(f'\t{v}')

    vars_der_no_izq = [nv for nv in bd_derecha.vars_base if nv not in bd_izquierda.vars_base]
    if vars_der_no_izq and imp:
        print(f'\nVariables en [{bd_derecha.nombre}] que no están en [{bd_izquierda.nombre}]:')
        for v in vars_der_no_izq:
            print(f'\t{v}')
    print()

    # Se obtienen las variables repetidas en ambas bases
    vars_repe = [nv for nv in bd_izquierda.vars_base if nv in bd_derecha.vars_base and nv not in identificador]

    if mantener == 'ambos':
        bd_izquierda.renombrar_var(vars_repe, lambda nvar, eti_o, ind: f"{nvar}_{suf_1}")
        bd_derecha.renombrar_var(vars_repe, lambda nvar, eti_o, ind: f"{nvar}_{suf_2}")
        bd_ret.df = pd.merge(bd_izquierda.df, bd_derecha.df, how='outer', on=identificador, validate='1:1',
                             suffixes=(f"_{suf_1}", f"_{suf_2}"))
    elif mantener == 'izq':
        bd_ret.df = pd.merge(bd_izquierda.df, bd_derecha.df.drop(vars_repe, axis=1), how='outer', on=identificador,
                             validate='1:1')
    elif mantener == 'der':
        bd_ret.df = pd.merge(bd_izquierda.df.drop(vars_repe, axis=1), bd_derecha.df, how='outer', on=identificador,
                             validate='1:1')

    bd_ret.df = bd_ret.df[bd_ret.vars_base]
    print(f'Base [{bd_ret.nombre}] creada')

    return bd_ret


def agregar_casos(bd_arriba: BaseDatos, bd_abajo: BaseDatos, identificador: [str, list, tuple],
                  verify_integrity: bool = True, nombre: str = None, imp: bool = True, mantener: str = 'arr',
                  suf_1: str = '1', suf_2: str = '2') -> BaseDatos:
    """
    Junta dos bases con identificadores distintos entre ambos
    Se valida que no haya ID repetidos
    """
    # Chequeos
    if bd_arriba.df.size == 0 and bd_abajo.df.size == 0:
        raise ErrBDD("Se está intentando juntar dos bases vacías")
    assert mantener in ('arr', 'aba', 'ambos'), "Error en el parámetro [mantener_metadata]"
    _mantener_metadata = '1' if mantener == 'arr' else '2' if mantener == 'aba' else mantener

    if isinstance(identificador, str):
        identificador = [identificador]
    bd_ret = juntar_metadata(bd_arriba, bd_abajo, nombre=nombre, mantener=_mantener_metadata, suf_1=suf_1, suf_2=suf_2,
                             ident=identificador)

    # Imprime las diferencias en las variables
    vars_arr_no_aba = [nv for nv in bd_arriba.vars_base if nv not in bd_abajo.vars_base]
    if vars_arr_no_aba and imp:
        print(f'\nVariables en [{bd_arriba.nombre}] que no están en [{bd_abajo.nombre}]:')
        for v in vars_arr_no_aba:
            print(f'\t{v}')
        print()

    vars_aba_no_arr = [nv for nv in bd_abajo.vars_base if nv not in bd_arriba.vars_base]
    if vars_aba_no_arr and imp:
        print(f'\nVariables en [{bd_abajo.nombre}] que no están en [{bd_arriba.nombre}]:')
        for v in vars_aba_no_arr:
            print(f'\t{v}')
        print()

    # Se guarda el orden de las variables
    orden_variables = bd_arriba.vars()

    # Esto se hace para que las variables de cadena que están en una y no en la otra no tengan datos espurios.
    df_abajo_tmp = bd_abajo.df.copy()
    for v in [v for v in vars_arr_no_aba if 'A' in bd_arriba.get_frmt(v)]:
        df_abajo_tmp[v] = ''

    df_arriba_tmp = bd_arriba.df.copy()
    for v in [v for v in vars_aba_no_arr if 'A' in bd_abajo.get_frmt(v)]:
        df_arriba_tmp[v] = ''

    # Aquí se toman los DataFrame de los objetos BaseDatos y se pone(n) la(s) variable(s) [on] como Index para
    # poder aplicar pd.concat y validar que no haya repetidos
    bd_ret.df = pd.concat([df_arriba_tmp.set_index(identificador, verify_integrity=verify_integrity),
                           df_abajo_tmp.set_index(identificador, verify_integrity=verify_integrity)],
                          verify_integrity=verify_integrity, )
    # Se regresa el Index a columna(s) y se regresa al orden original
    bd_ret.df = bd_ret.df.copy()
    bd_ret.df.reset_index(inplace=True)
    bd_ret.incluir_vars(orden_variables, todas=True)
    bd_ret.df = bd_ret.df[bd_ret.vars_base]

    print(f'Base [{bd_ret.nombre}] creada')

    return bd_ret

# def reestructurar_car(bd: BaseDatos, d_rest: dict, nom_col: str, eti_col: str = None, eti_vals_col: dict = None) -> BaseDatos:
#     """
#     Reestructura la base de columnas a renglones
#     """
#     bd.df.melt()
