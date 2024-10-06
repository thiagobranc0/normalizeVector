#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <xmmintrin.h> // Para SSE
#define _GNU_SOURCE

#define LOOKUP_TABLE_SIZE 10000

// Função naïve para normalizar um vetor de características
void normalize_feature_vector(float *features, int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    sum += features[i] * features[i];
  }
  float inv_sqrt = 1.0f / sqrt(sum);

  for (int i = 0; i < length; i++) {
    features[i] *= inv_sqrt;
  }
}

// Função de normalização usando o método de Quake III
float quake3_rsqrt(float number) {
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;           // Obtenha o valor inteiro de y
  i = 0x5f3759df - (i >> 1); // Aplica a constante mágica e deslocamento
  y = *(float *)&i;          // Converte o valor de volta para float
  y = y * (threehalfs - (x2 * y * y)); // Iteração de Newton

  return y;
}

void quake3_normalize_feature_vector(float *features, int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    sum += features[i] * features[i];
  }
  float inv_sqrt = quake3_rsqrt(sum);

  for (int i = 0; i < length; i++) {
    features[i] *= inv_sqrt;
  }
}

// Função de normalização usando SSE
void sse_normalize_feature_vector(float *features, int length) {
  __m128 sum_vec = _mm_setzero_ps();

  // Soma dos quadrados dos elementos
  for (int i = 0; i < length; i += 4) {
    __m128 vec = _mm_loadu_ps(&features[i]);
    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(vec, vec));
  }

  // Soma horizontal dos 4 valores no vetor sum_vec
  float sum_arr[4];
  _mm_storeu_ps(sum_arr, sum_vec);
  float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

  // Inversão de raiz quadrada
  __m128 inv_sqrt_vec = _mm_rsqrt_ps(_mm_set1_ps(sum));

  // Normaliza o vetor
  for (int i = 0; i < length; i += 4) {
    __m128 vec = _mm_loadu_ps(&features[i]);
    vec = _mm_mul_ps(vec, inv_sqrt_vec);
    _mm_storeu_ps(&features[i], vec);
  }
}

// Tabela de consulta para armazenar as inversões de raiz quadrada
float lookup_table[LOOKUP_TABLE_SIZE];

// Função para inicializar a tabela de consulta
void init_lookup_table() {
  for (int i = 0; i < LOOKUP_TABLE_SIZE; i++) {
    float x = (float)i / (LOOKUP_TABLE_SIZE - 1);
    lookup_table[i] = 1.0f / sqrt(x);
  }
}

// Função para obter a inversão de raiz quadrada da tabela de consulta
float lookup_rsqrt(float number) {
  int index = (int)(number * (LOOKUP_TABLE_SIZE - 1));
  if (index < 0)
    index = 0;
  if (index >= LOOKUP_TABLE_SIZE)
    index = LOOKUP_TABLE_SIZE - 1;
  return lookup_table[index];
}

void lookup_table_normalize_feature_vector(float *features, int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    sum += features[i] * features[i];
  }

  float inv_sqrt = lookup_rsqrt(sum);

  for (int i = 0; i < length; i++) {
    features[i] *= inv_sqrt;
  }
}

// Função para ler dados de um arquivo CSV
float **read_csv(const char *filename, int *num_elements, int *num_dimensions) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  // Determine the number of elements and dimensions
  *num_elements = 0;
  *num_dimensions = 0;
  char line[1024];
  while (fgets(line, sizeof(line), file)) {
    if (*num_elements == 0) {
      char *token = strtok(line, ",");
      while (token) {
        (*num_dimensions)++;
        token = strtok(NULL, ",");
      }
    }
    (*num_elements)++;
  }
  rewind(file);

  // Allocate memory for the features
  float **features = (float **)malloc(*num_elements * sizeof(float *));
  for (int i = 0; i < *num_elements; i++) {
    features[i] = (float *)malloc(*num_dimensions * sizeof(float));
  }

  // Read the data
  int i = 0;
  while (fgets(line, sizeof(line), file)) {
    int j = 0;
    char *token = strtok(line, ",");
    while (token) {
      features[i][j++] = atof(token);
      token = strtok(NULL, ",");
    }
    i++;
  }

  fclose(file);
  return features;
}

// Função para medir o tempo de execução usando a biblioteca 'resources'
void get_resource_usage(struct rusage *usage) { getrusage(RUSAGE_SELF, usage); }

void print_resource_usage(const char *label, struct rusage *usage) {
  printf("%s\n", label);
  printf("User time: %ld.%06ld seconds\n", usage->ru_utime.tv_sec,
         usage->ru_utime.tv_usec);
  printf("System time: %ld.%06ld seconds\n", usage->ru_stime.tv_sec,
         usage->ru_stime.tv_usec);
  printf("Maximum resident set size: %ld kilobytes\n", usage->ru_maxrss);
}

int main() {
  int num_elements, num_dimensions;
  float **features = read_csv("data.csv", &num_elements, &num_dimensions);

  // Inicializa a tabela de consulta
  init_lookup_table();

  struct rusage start_usage, end_usage;

  get_resource_usage(&start_usage);

  // Só tirar os comentários da função que deseja usar:
  for (int i = 0; i < num_elements; i++) {
    // normalize_feature_vector(features[i], num_dimensions);

    quake3_normalize_feature_vector(features[i], num_dimensions);
    // sse_normalize_feature_vector(features[i], num_dimensions);
    // lookup_table_normalize_feature_vector(features[i],num_dimensions);
  }

  get_resource_usage(&end_usage);

  printf("Normalized features:\n");
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < num_dimensions; j++) {
      printf("%f ", features[i][j]);
    }
    printf("\n");
  }

  printf("Execution time and resource usage:\n");
  print_resource_usage("Start Usage", &start_usage);
  print_resource_usage("End Usage", &end_usage);

  // Free allocated memory
  for (int i = 0; i < num_elements; i++) {
    free(features[i]);
  }
  free(features);

  return 0;
}
