from ortools.linear_solver import pywraplp
from math import ceil
from random import randint
import json
import os

def newSolver(name,integer=False):
  return pywraplp.Solver(name,pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def SolVal(x):
  if type(x) is not list:
    return 0 if x is None \
      else x if isinstance(x,(int,float)) \
           else x.SolutionValue() if x.Integer() is False \
                else int(x.SolutionValue())
  elif type(x) is list:
    return [SolVal(e) for e in x]

def ObjVal(x):
  return x.Objective().Value()


def solve_model(demands, parent_width=100):
  num_orders = len(demands)
  solver = newSolver('Cutting Stock', True)
  k,b  = bounds(demands, parent_width)
  y = [ solver.IntVar(0, 1, f'y_{i}') for i in range(k[1]) ] 
  x = [[solver.IntVar(0, b[i], f'x_{i}_{j}') for j in range(k[1])] \
      for i in range(num_orders)]
  
  unused_widths = [ solver.NumVar(0, parent_width, f'w_{j}') \
      for j in range(k[1]) ] 
  
  nb = solver.IntVar(k[0], k[1], 'nb')

  for i in range(num_orders):  
    solver.Add(sum(x[i][j] for j in range(k[1])) >= demands[i][0]) 

  for j in range(k[1]):
    solver.Add( \
        sum(demands[i][1]*x[i][j] for i in range(num_orders)) \
        <= parent_width*y[j] \
      ) 

    solver.Add(parent_width*y[j] - sum(demands[i][1]*x[i][j] for i in range(num_orders)) == unused_widths[j])


    if j < k[1]-1: 
      solver.Add(sum(x[i][j] for i in range(num_orders)) >= sum(x[i][j+1] for i in range(num_orders)))

  solver.Add(nb == solver.Sum(y[j] for j in range(k[1])))

  Cost = solver.Sum((j+1)*y[j] for j in range(k[1]))

  solver.Minimize(Cost)

  status = solver.Solve()
  numRollsUsed = SolVal(nb)

  return status, \
    numRollsUsed, \
    rolls(numRollsUsed, SolVal(x), SolVal(unused_widths), demands), \
    SolVal(unused_widths), \
    solver.WallTime()

def bounds(demands, parent_width=100):
  num_orders = len(demands)
  b = []
  T = 0
  k = [0,1]
  TT = 0

  for i in range(num_orders):
    quantity, width = demands[i][0], demands[i][1]
    b.append( min(quantity, int(round(parent_width / width))) )

    if T + quantity*width <= parent_width:
      T, TT = T + quantity*width, TT + quantity*width

    else:
      while quantity:
        if T + width <= parent_width:
          T, TT, quantity = T + width, TT + width, quantity-1
        else:
          k[1],T = k[1]+1, 0 
  k[0] = int(round(TT/parent_width+0.5))


  return k, b


def rolls(nb, x, w, demands):
  consumed_big_rolls = []
  num_orders = len(x) 
  for j in range(len(x[0])):
    RR = [ abs(w[j])] + [ int(x[i][j])*[demands[i][1]] for i in range(num_orders) \
                    if x[i][j] > 0 ] 
    consumed_big_rolls.append(RR)

  return consumed_big_rolls



def solve_large_model(demands, parent_width=100):
  num_orders = len(demands)
  iter = 0
  patterns = get_initial_patterns(demands)
  quantities = [demands[i][0] for i in range(num_orders)]
  # print('quantities', quantities)

  while iter < 20:
    status, y, l = solve_master(patterns, quantities, parent_width=parent_width)
    iter += 1

    widths = [demands[i][1] for i in range(num_orders)]
    new_pattern, objectiveValue = get_new_pattern(l, widths, parent_width=parent_width)


    for i in range(num_orders):
      patterns[i].append(new_pattern[i])

  status, y, l = solve_master(patterns, quantities, parent_width=parent_width, integer=True)  

  return status, \
          patterns, \
          y, \
          rolls_patterns(patterns, y, demands, parent_width=parent_width)


def solve_master(patterns, quantities, parent_width=100, integer=False):
  title = 'Cutting stock master problem'
  num_patterns = len(patterns)
  n = len(patterns[0])

  constraints = []

  solver = newSolver(title, integer)

  y = [ solver.IntVar(0, 1000, '') for j in range(n) ] 
  Cost = sum(y[j] for j in range(n)) 
  solver.Minimize(Cost)

  for i in range(num_patterns):
    constraints.append(solver.Add( sum(patterns[i][j]*y[j] for j in range(n)) >= quantities[i]) ) 

  status = solver.Solve()
  y = [int(ceil(e.SolutionValue())) for e in y]

  l =  [0 if integer else constraints[i].DualValue() for i in range(num_patterns)]
  toreturn = status, y, l
  return toreturn

def get_new_pattern(l, w, parent_width=100):
  solver = newSolver('Cutting stock sub-problem', True)
  n = len(l)
  new_pattern = [ solver.IntVar(0, parent_width, '') for i in range(n) ]


  Cost = sum( l[i] * new_pattern[i] for i in range(n))
  solver.Maximize(Cost)
  solver.Add( sum( w[i] * new_pattern[i] for i in range(n)) <= parent_width ) 

  status = solver.Solve()
  return SolVal(new_pattern), ObjVal(solver)

def get_initial_patterns(demands):
  num_orders = len(demands)
  return [[0 if j != i else 1 for j in range(num_orders)]\
          for i in range(num_orders)]

def rolls_patterns(patterns, y, demands, parent_width=100):
  R, m, n = [], len(patterns), len(y)

  for j in range(n):
    for _ in range(y[j]):
      RR = []
      for i in range(m):
        if patterns[i][j] > 0:
          RR.extend( [demands[i][1]] * int(patterns[i][j]) )
      used_width = sum(RR)
      R.append([parent_width - used_width, RR])

  return R


def checkWidths(demands, parent_width):
  for quantity, width in demands:
    if width > parent_width:
      print(f'Требуемый размер балки {width} больше размера родительской балки {parent_width}')
      return False
  return True


def StockCutter1D(child_rolls, parent_rolls, output_json=True, large_model=True):
  parent_width = parent_rolls[0][1]

  if not checkWidths(demands=child_rolls, parent_width=parent_width):
    return []


  # print('child_rolls', child_rolls)
  # print('parent_rolls', parent_rolls)

  if not large_model:
    status, numRollsUsed, consumed_big_rolls, unused_roll_widths, wall_time = \
              solve_model(demands=child_rolls, parent_width=parent_width)

    new_consumed_big_rolls = []
    for big_roll in consumed_big_rolls:
      if len(big_roll) < 2:
        consumed_big_rolls.remove(big_roll)
        continue
      unused_width = big_roll[0]
      subrolls = []
      for subitem in big_roll[1:]:
        if isinstance(subitem, list):
          subrolls = subrolls + subitem
        else:
          subrolls.append(subitem)
      new_consumed_big_rolls.append([unused_width, subrolls])
    consumed_big_rolls = new_consumed_big_rolls
  else:
    status, A, y, consumed_big_rolls = solve_large_model(demands=child_rolls, parent_width=parent_width)

  numRollsUsed = len(consumed_big_rolls)



  STATUS_NAME = ['OPTIMAL',
    'FEASIBLE',
    'INFEASIBLE',
    'UNBOUNDED',
    'ABNORMAL',
    'NOT_SOLVED'
    ]

  output = {
      "statusName": STATUS_NAME[status],
      "numSolutions": '1',
      "numUniqueSolutions": '1',
      "numRollsUsed": numRollsUsed,
      "solutions": consumed_big_rolls 
  }

  print('Количество родительских балок', numRollsUsed)
  print('Статус:', output['statusName'])
  print('Найденные решения :', output['numSolutions'])
  print('Уникальные решения : ', output['numUniqueSolutions'])

  if output_json:
    return json.dumps(output)        
  else:
    return consumed_big_rolls

def main():
  print("Введите размер родительской балки")
  parent = int(input())
  parent_rolls = [[10, parent]]
  print("< Для остановки ввода данных нажмите q >")
  child_rolls = []
  while True:
    print("Введите размеры заготовки, которую вы хотите получить: ")
    size = input()
    if size == 'q':
      break
    size = int(size)
    print("Введите количество заготовок, которые вы хотите получить: ")
    quant = input()
    if quant == 'q':
      break
    quant = int(quant)
    child_rolls.append([quant, size])

  if len(child_rolls) > 0:
    consumed_big_rolls = StockCutter1D(child_rolls, parent_rolls, output_json=False, large_model=False)

    for idx, roll in enumerate(consumed_big_rolls):
      print(f"Шаблон для разреза балки #{idx}:{sorted(roll[1], reverse=True)}")
    print("Сохранить результат в текстовый файл? y / n")
    cmd = input()
    if cmd == 'y':
      with open("result.txt", 'a', encoding="UTF-8") as out:
        print(f"Для родительской балки длины {parent} и необходимых заготовок количеств/длин {child_rolls}", file=out)
        for idx, roll in enumerate(consumed_big_rolls):
          print(f"Шаблон для разреза балки #{idx + 1}:{sorted(roll[1], reverse=True)}", file=out)
        print('\n', file=out)
  else:
    print("Данные для заготовок не были введены!")

if __name__ == '__main__':
  with open("result.txt", 'w', encoding="UTF-8") as f:
    # print('', file=f)
    pass
  while True:
    main()
    print("Хотите воспользоваться программой ещё раз? y / n ")
    cmd = input()
    if cmd == 'y':
      continue
    elif cmd == 'n':
      break


  os.system("pause")
  



