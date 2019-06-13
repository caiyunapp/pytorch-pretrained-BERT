from django.shortcuts import render
# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import HttpResponse
import json
import sys
sys.path =['/home/xd/projects/pytorch-pretrained-BERT'] + sys.path
from likunlin_final import analyze_text,modify_text

text = []
def home(request):
    return render(request, 'home.html')


def analyse(request):
    global text
    text = request.GET['text']
    text = [text]
    print("xiaofang")
    suggestions,tokens,avg_gap = analyze_text(text)
    return HttpResponse(json.dumps({"tokens":tokens,"suggestions":suggestions,"avg_gap":avg_gap}))

def modify(request): 
    global text
    index = request.GET['index']
    text,new_tokens,suggestions = modify_text(int(index),text)
    return HttpResponse(json.dumps({"tokens":new_tokens,"suggestions":suggestions}))
