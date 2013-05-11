task void select_ispc_task( uniform int rowsPerTask,
							uniform int input[],
							uniform int output[])
{
	uniform int rowStart = taskIndex * rowsPerTask;
	uniform int rowStop = rowStart +  rowsPerTask;
    uniform int divider = 2;
	foreach(i = rowStart ... rowStop) {
		if (input[i] > 500 ) {
            output[i] = input[i];
        }
	}
}
export void select_ispc_withtasks( uniform int N,
								 uniform int input[],
							     uniform int output[])
{
	uniform int rowsPerTask = N / 4;
	launch[4] select_ispc_task(rowsPerTask, input, output);
}

static int warp_scan(int value)
{
  for (int i = 1; i < programCount; i *= 2)
  {
    int n = shuffle(value, max(programIndex-i,0));
    if (programIndex >= i)
      value += n;
  }
  return value;
}

static int warp_scan16(int value)
{
  int n, i;
  i = 1; value += shuffle(value, programIndex-i) & (-(int)(programIndex >= i));
  i = 2; value += shuffle(value, programIndex-i) & (-(int)(programIndex >= i));
  i = 4; value += shuffle(value, programIndex-i) & (-(int)(programIndex >= i));
  if (programCount >= 8)
  {
    i = 8; value += shuffle(value, programIndex-i) & (-(int)(programIndex >= i));
    if (programCount >= 16) 
      i = 16; value += shuffle(value, programIndex-i) & (-(int)(programIndex >= i));
  }
  return value;
}

export void exclusive_scan(
    uniform const int n,
    uniform       int data[])
{
  int carryValue = 0;
  int x;
  int y = 0;
  foreach(i=0...n)
  {
    carryValue += extract(y,programCount-1);
    x = data[i];
    y = warp_scan16(x);
    data[i] = y + carryValue - x;
  }
}

export void add(
    uniform const int n,
    uniform const int value,
    uniform       int data[])
{
  foreach(i=0...n)
  {
    data[i] += value;
  }
}