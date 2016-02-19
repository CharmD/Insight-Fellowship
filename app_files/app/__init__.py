# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:40:06 2016

@author: Charm

Creating flask

"""

from flask import Flask

app = Flask(__name__) # Flask is a class

from app import views