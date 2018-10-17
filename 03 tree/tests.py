import unittest
import numpy as np
import tree


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.obj_tree = tree.DecisionTree()

    def test_entropy(self):
        test_arr = []
        test_arr.append({'arr': np.array([1 for x in range(9)] + [0 for x in range(11)]), 
        	             'answer':0.993})
        test_arr.append({'arr': np.array([1 for x in range(8)] + [0 for x in range(5)]), 
        	             'answer':0.961})
        test_arr.append({'arr': np.array([1 for x in range(1)] + [0 for x in range(6)]), 
        	             'answer':0.592})
        test_arr.append({'arr': np.array([1, 2, 3, 4, 5, 6]), 
        	             'answer':2.585})
        for x in test_arr:
        	entropy = self.obj_tree._entropy(x['arr'])
        	self.assertEqual(round(entropy, 3), x['answer'])


if __name__ == '__main__':
    unittest.main()


'''
Описание модуля unittest:
    https://pythonworld.ru/moduli/modul-unittest.html

Специальные функци:
    def setUp(self):
        # Код настройки - перед запуском тестов
        pass
    def tearDown(self):
        # Код после выполнения тестов
        pass

Пропуск тестов:
    @unittest.skip(reason) - пропустить тест. reason описывает причину пропуска.
    @unittest.skipIf(condition, reason) - пропустить тест, если condition истинно.
    @unittest.skipUnless(condition, reason) - пропустить тест, если condition ложно.
    @unittest.expectedFailure - пометить тест как ожидаемая ошибка.

Функции для проверок:
    assertEqual(a, b) — a == b
    assertNotEqual(a, b) — a != b
    assertTrue(x) — bool(x) is True
    assertFalse(x) — bool(x) is False
    assertIs(a, b) — a is b
    assertIsNot(a, b) — a is not b
    assertIsNone(x) — x is None
    assertIsNotNone(x) — x is not None
    assertIn(a, b) — a in b
    assertNotIn(a, b) — a not in b
    assertIsInstance(a, b) — isinstance(a, b)
    assertNotIsInstance(a, b) — not isinstance(a, b)
    assertRaises(exc, fun, *args, **kwds) — fun(*args, **kwds) порождает исключение exc
    assertRaisesRegex(exc, r, fun, *args, **kwds) — fun(*args, **kwds) порождает исключение exc и сообщение соответствует регулярному выражению r
    assertWarns(warn, fun, *args, **kwds) — fun(*args, **kwds) порождает предупреждение
    assertWarnsRegex(warn, r, fun, *args, **kwds) — fun(*args, **kwds) порождает предупреждение и сообщение соответствует регулярному выражению r
    assertAlmostEqual(a, b) — round(a-b, 7) == 0
    assertNotAlmostEqual(a, b) — round(a-b, 7) != 0
    assertGreater(a, b) — a > b
    assertGreaterEqual(a, b) — a >= b
    assertLess(a, b) — a < b
    assertLessEqual(a, b) — a <= b
    assertRegex(s, r) — r.search(s)
    assertNotRegex(s, r) — not r.search(s)
    assertCountEqual(a, b) — a и b содержат те же элементы в одинаковых количествах, но порядок не важен

Запуск теста:
	1. Запустив на исполнение файл с тестом
	2. Запустив команду в консоли: python -m unittest

Дополнительные настроийки запуска из консоли:
	-b (--buffer) - вывод программы при провале теста будет показан, а не скрыт, как обычно.
	-c (--catch) - Ctrl+C во время выполнения теста ожидает завершения текущего теста и затем сообщает результаты на данный момент. Второе нажатие Ctrl+C вызывает обычное исключение KeyboardInterrupt.
	-f (--failfast) - выход после первого же неудачного теста.
	--locals (начиная с Python 3.5) - показывать локальные переменные для провалившихся тестов.
'''